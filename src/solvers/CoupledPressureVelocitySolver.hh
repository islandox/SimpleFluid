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
#include "FVM/FaceFlux.hh"
#include "FVM/OperatorDetails.hh"
#include "solvers/BelosLinearSolver.hh"

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Ifpack2_Factory.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <TpetraExt_MatrixMatrix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
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
 * Pressure blocks act on internally normalized pressure p/rho_ref; public
 * solver fields are converted to and from Pa at the solve boundary.
 *
 * @tparam Pack Tpetra type pack providing matrix, map, and vector types.
 */
template<TpetraTypePack Pack>
struct CoupledPressureVelocitySystem
{
    using matrix_type = typename Pack::matrix_type;
    using map_type = typename Pack::map_type;
    using scalar_type = typename Pack::scalar_type;
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
    scalar_type reference_density = scalar_type{1};
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

/**
 * @brief Controls when coupled operator and solver setup is rebuilt.
 *
 * Immutable mesh maps and boundary-face locations are always retained for the
 * solver lifetime. `OnOperatorGraphChange` additionally retains compatible
 * matrix graphs, Schur products, preconditioners, and Belos manager state.
 */
enum class CoupledRebuildPolicy : std::uint8_t
{
    OnOperatorGraphChange = 0, ///< Reuse compatible graphs and numeric state.
    Always                 = 1  ///< Rebuild operator-dependent setup each call.
};

/**
 * @brief Observable counters for coupled setup reuse.
 *
 * The counters are intended for regression tests and lightweight runtime
 * instrumentation. They do not participate in the numerical algorithm.
 */
struct CoupledPressureVelocityCacheStatistics
{
    size_t coupled_map_builds = 0;
    size_t static_geometry_builds = 0;
    size_t static_geometry_reuses = 0;
    size_t matrix_graph_reuses = 0;
    size_t schur_product_reuses = 0;
    size_t preconditioner_builds = 0;
    size_t preconditioner_numeric_reuses = 0;
    size_t belos_solver_builds = 0;
    size_t belos_solver_reuses = 0;
    size_t preconditioner_scratch_allocations = 0;
};

namespace detail
{

/**
 * @brief Accumulate one sparse coefficient into an assembly row.
 *
 * @tparam Column Sparse column-index type.
 * @tparam Scalar Sparse coefficient type.
 * @param row Row accumulator keyed by column.
 * @param column Column receiving the contribution.
 * @param value Contribution to add.
 */
template<class Column, class Scalar>
void add_entry(std::unordered_map<Column, Scalar>& row,
               Column column,
               Scalar value)
{
    row[column] += value;
}

/**
 * @brief Build the four-unknown-per-cell map for a coupled system.
 *
 * @tparam Pack Tpetra type pack defining the distributed map.
 * @param mesh Mesh that owns the cell distribution.
 * @return Coupled velocity-pressure map.
 */
template<TpetraTypePack Pack>
auto make_coupled_map(const Mesh<Pack>& mesh)
    -> Teuchos::RCP<const typename Pack::map_type>
{
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;

    constexpr global_ordinal_type block_size = 4;
    const auto invalid_global_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const auto comm = mesh.owned_cell_map()->getComm();

    return Teuchos::rcp(new map_type(
        invalid_global_size,
        block_size * mesh.num_owned_cells(),
        global_ordinal_type{0},
        comm));
}

/** @brief Matrix reset for cached-graph assembly. */
template<TpetraTypePack Pack>
struct PreparedCoupledMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
};

/**
 * @brief Allocate a matrix or reset a compatible cached graph.
 */
template<TpetraTypePack Pack>
PreparedCoupledMatrix<Pack> prepare_coupled_matrix(
    const Teuchos::RCP<const typename Pack::map_type>& row_map,
    const Teuchos::RCP<const typename Pack::map_type>& column_map,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    size_t entries_per_row)
{
    using matrix_type = typename Pack::matrix_type;

    const auto compatible =
        !cached_matrix.is_null()
        && cached_matrix->isFillComplete()
        && cached_matrix->getRowMap()->isSameAs(*row_map)
        && cached_matrix->getDomainMap()->isSameAs(*row_map)
        && (column_map.is_null()
            || (!cached_matrix->getColMap().is_null()
                && cached_matrix->getColMap()->isSameAs(*column_map)));
    if (!compatible)
    {
        if (column_map.is_null())
        {
            return {
                Teuchos::rcp(new matrix_type(row_map, entries_per_row)),
                false};
        }
        return {
            Teuchos::rcp(new matrix_type(
                row_map, column_map, entries_per_row)),
            false};
    }

    cached_matrix->resumeFill();
    cached_matrix->setAllToScalar(typename Pack::scalar_type{});
    return {std::move(cached_matrix), true};
}

/** @brief Insert into a fresh local graph or sum into a reused one. */
template<TpetraTypePack Pack>
void add_coupled_local_values(
    const PreparedCoupledMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::local_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values)
{
    if (!prepared.reused)
    {
        prepared.matrix->insertLocalValues(row, columns, values);
        return;
    }
    const auto updated = prepared.matrix->sumIntoLocalValues(
        row, columns, values);
    if (updated != static_cast<typename Pack::local_ordinal_type>(
                       columns.size()))
    {
        throw std::invalid_argument(
            "Coupled cached matrix graph is incompatible with local assembly.");
    }
}

/** @brief Insert into a fresh global graph or sum into a reused one. */
template<TpetraTypePack Pack>
void add_coupled_global_values(
    const PreparedCoupledMatrix<Pack>& prepared,
    typename Pack::global_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::global_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values)
{
    if (!prepared.reused)
    {
        prepared.matrix->insertGlobalValues(row, columns, values);
        return;
    }
    const auto updated = prepared.matrix->sumIntoGlobalValues(
        row, columns, values);
    if (updated != static_cast<typename Pack::local_ordinal_type>(
                       columns.size()))
    {
        throw std::invalid_argument(
            "Coupled cached matrix graph is incompatible with global assembly.");
    }
}

/**
 * @brief Affine least-squares stencil for a normalized pressure gradient.
 *
 * @tparam Pack Tpetra type pack defining mesh ordinals and vectors.
 */
template<TpetraTypePack Pack>
struct AffinePressureGradientStencil
{
    FVM::detail::LeastSquaresGradientStencil<Mesh<Pack>> entries;
    typename Mesh<Pack>::Vec3 constant{};
};

/**
 * @brief Linearize the boundary-aware least-squares pressure gradient.
 *
 * The coupled matrix acts on normalized pressure q = p/rho_ref. Cell
 * coefficients populate its pressure block, while the constant vector
 * carries prescribed Dirichlet values and Neumann gradients to the RHS.
 *
 * @tparam Pack Tpetra type pack defining mesh and scalar types.
 * @param mesh Mesh providing cells, faces, and boundary geometry.
 * @param boundary_conditions Pressure boundary conditions by patch name.
 * @param reference_density Density used to normalize physical pressure.
 * @return One affine pressure-gradient stencil per owned cell.
 * @throws std::invalid_argument for unsupported pressure boundary types.
 */
template<TpetraTypePack Pack>
auto pressure_gradient_stencils(
    const Mesh<Pack>& mesh,
    const BoundaryConditionMap& boundary_conditions,
    typename Pack::scalar_type reference_density)
    -> std::vector<AffinePressureGradientStencil<Pack>>
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = Mesh<Pack>;

    std::vector<AffinePressureGradientStencil<Pack>> stencils(
        mesh.num_owned_cells());
    const auto boundary_locations =
        FVM::detail::boundary_face_locations(mesh);
    auto boundary_condition =
        [&](local_ordinal_type face_lid)
            -> std::optional<BoundaryCondition>
    {
        if (!mesh.is_boundary_face(face_lid)
            || static_cast<size_t>(face_lid)
               >= boundary_locations.size())
        {
            return std::nullopt;
        }
        const auto location =
            boundary_locations[static_cast<size_t>(face_lid)];
        if (!location.active)
        {
            return std::nullopt;
        }
        const auto iter = boundary_conditions.find(
            mesh.boundary_batch_name(location.batch_id));
        return iter == boundary_conditions.end()
             ? BoundaryCondition{}
             : iter->second;
    };

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        std::array<std::array<real_t, 3>, 3> normal{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            typename mesh_type::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                direction =
                    mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (boundary_condition(face_lid))
            {
                direction = mesh.face_centroid(face_lid)
                          - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }

            normal[0][0] += direction.x * direction.x;
            normal[0][1] += direction.x * direction.y;
            normal[0][2] += direction.x * direction.z;
            normal[1][1] += direction.y * direction.y;
            normal[1][2] += direction.y * direction.z;
            normal[2][2] += direction.z * direction.z;
        }
        normal[1][0] = normal[0][1];
        normal[2][0] = normal[0][2];
        normal[2][1] = normal[1][2];

        std::unordered_map<local_ordinal_type,
                           typename mesh_type::Vec3> coefficients;
        typename mesh_type::Vec3 constant{};
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            typename mesh_type::Vec3 direction{};
            std::optional<BoundaryCondition> condition;
            if (mesh.is_interior_face(face_lid))
            {
                direction =
                    mesh.cell_center_vector(face_lid, cell_lid);
            }
            else
            {
                condition = boundary_condition(face_lid);
                if (!condition)
                {
                    continue;
                }
                direction = mesh.face_centroid(face_lid)
                          - mesh.cell_centroid(cell_lid);
            }

            auto local_normal = normal;
            const auto basis =
                FVM::detail::solve_3x3(local_normal, direction);
            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                FVM::detail::add_gradient_coefficient<mesh_type>(
                    coefficients, other, basis);
                FVM::detail::add_gradient_coefficient<mesh_type>(
                    coefficients,
                    cell_lid,
                    {-basis.x, -basis.y, -basis.z});
                continue;
            }

            if (condition->type == BoundaryConditionType::Dirichlet)
            {
                FVM::detail::add_gradient_coefficient<mesh_type>(
                    coefficients,
                    cell_lid,
                    {-basis.x, -basis.y, -basis.z});
                const auto normalized_value =
                    condition->value / reference_density;
                constant = constant + basis * normalized_value;
            }
            else if (condition->type
                     == BoundaryConditionType::Neumann)
            {
                const auto normalized_delta =
                    condition->value / reference_density
                  * FVM::detail::boundary_normal_distance(
                        mesh, face_lid, cell_lid);
                constant = constant + basis * normalized_delta;
            }
            else
            {
                throw std::invalid_argument(
                    "Coupled pressure gradient supports only Dirichlet "
                    "and Neumann boundary conditions.");
            }
        }

        auto& stencil = stencils[owned];
        stencil.entries.reserve(coefficients.size());
        for (const auto& [entry_lid, coefficient] : coefficients)
        {
            stencil.entries.push_back({entry_lid, coefficient});
        }
        stencil.constant = constant;
    }
    return stencils;
}

/**
 * @brief Left-scale a gradient matrix by an inverse momentum diagonal.
 *
 * @tparam Pack Tpetra type pack defining matrix and vector types.
 * @param gradient Gradient operator to scale.
 * @param inverse_diagonal Per-row inverse momentum diagonal.
 * @param cached_matrix Optional compatible matrix whose graph is reused.
 * @return Scaled gradient matrix with refreshed numeric values.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
scaled_gradient_matrix(
    const typename Pack::matrix_type& gradient,
    const typename Pack::vector_type& inverse_diagonal,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;

    const auto prepared = prepare_coupled_matrix<Pack>(
        gradient.getRowMap(), gradient.getColMap(),
        std::move(cached_matrix),
        gradient.getLocalMaxNumRowEntries());
    const auto& scaled = prepared.matrix;
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
        add_coupled_local_values<Pack>(
            prepared, static_cast<local_ordinal_type>(row),
            copied_columns(), copied_values());
    }
    scaled->fillComplete();
    return scaled;
}

/** @brief Persistent scratch matrices for Schur numeric updates. */
template<TpetraTypePack Pack>
struct CoupledSchurWorkspace
{
    Teuchos::RCP<typename Pack::vector_type> inverse_diagonal;
    std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>
        scaled_gradient;
    std::array<Teuchos::RCP<typename Pack::matrix_type>, 3> product;

    void clear()
    {
        inverse_diagonal = Teuchos::null;
        scaled_gradient = {};
        product = {};
    }
};

/**
 * @brief Build the pressure Schur-complement approximation.
 *
 * @tparam Pack Tpetra type pack defining matrix and ordinal types.
 * @param momentum Momentum block used for diagonal inversion.
 * @param gradient Cartesian pressure-gradient blocks.
 * @param divergence Cartesian velocity-divergence blocks.
 * @param pressure_stabilization Pressure stabilization block.
 * @param pressure_gauge_gid Optional pressure row fixed as the gauge.
 * @param cached_schur Optional compatible Schur matrix to refresh.
 * @param workspace Optional persistent scaled-gradient and product storage.
 * @param reused_products Optional output set when every product was reused.
 * @return Assembled Schur-complement approximation.
 * @throws std::runtime_error if the momentum diagonal is singular.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
build_schur_approximation(
    const typename Pack::matrix_type& momentum,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& gradient,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& divergence,
    const typename Pack::matrix_type& pressure_stabilization,
    std::optional<typename Pack::global_ordinal_type>
        pressure_gauge_gid,
    Teuchos::RCP<typename Pack::matrix_type> cached_schur = Teuchos::null,
    CoupledSchurWorkspace<Pack>* workspace = nullptr,
    bool* reused_products = nullptr)
{
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using vector_type = typename Pack::vector_type;

    CoupledSchurWorkspace<Pack> local_workspace;
    auto& active_workspace =
        workspace == nullptr ? local_workspace : *workspace;
    if (active_workspace.inverse_diagonal.is_null()
        || !active_workspace.inverse_diagonal->getMap()->isSameAs(
            *momentum.getRowMap()))
    {
        active_workspace.inverse_diagonal = Teuchos::rcp(
            new vector_type(momentum.getRowMap(), true));
    }
    auto& inverse_diagonal = *active_workspace.inverse_diagonal;
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

    bool products_reused = true;
    for (size_t component = 0; component < gradient.size(); ++component)
    {
        const auto had_scaled =
            !active_workspace.scaled_gradient[component].is_null();
        auto scaled = scaled_gradient_matrix<Pack>(
            *gradient[component], inverse_diagonal,
            active_workspace.scaled_gradient[component]);
        active_workspace.scaled_gradient[component] = scaled;

        auto& product = active_workspace.product[component];
        const auto had_product = !product.is_null();
        if (product.is_null())
        {
            product = Teuchos::rcp(new matrix_type(
                divergence[component]->getRowMap(), 32));
        }
        Tpetra::MatrixMatrix::Multiply(
            *divergence[component], false,
            *scaled, false,
            *product, true);
        products_reused = products_reused && had_scaled && had_product;
    }
    if (reused_products != nullptr)
    {
        *reused_products = products_reused;
    }

    const auto prepared_schur = prepare_coupled_matrix<Pack>(
        pressure_stabilization.getRowMap(), Teuchos::null,
        std::move(cached_schur), 32);
    const auto& schur = prepared_schur.matrix;
    const auto stabilization_col_map =
        pressure_stabilization.getColMap();
    for (size_t row = 0;
         row < pressure_stabilization.getLocalNumRows();
         ++row)
    {
        const auto local_row = static_cast<local_ordinal_type>(row);
        const auto global_row =
            pressure_stabilization.getRowMap()->getGlobalElement(local_row);
        if (pressure_gauge_gid
            && global_row == *pressure_gauge_gid)
        {
            Teuchos::Array<global_ordinal_type> columns{global_row};
            Teuchos::Array<scalar_type> values{scalar_type{1}};
            add_coupled_global_values<Pack>(
                prepared_schur, global_row, columns(), values());
            continue;
        }

        std::unordered_map<global_ordinal_type, scalar_type> row_values;
        typename matrix_type::local_inds_host_view_type local_columns;
        typename matrix_type::values_host_view_type local_values;
        for (const auto& product : active_workspace.product)
        {
            product->getLocalRowView(
                local_row, local_columns, local_values);
            const auto product_col_map = product->getColMap();
            for (size_t entry = 0;
                 entry < local_columns.extent(0);
                 ++entry)
            {
                row_values[product_col_map->getGlobalElement(
                    local_columns[entry])] -= local_values[entry];
            }
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
        add_coupled_global_values<Pack>(
            prepared_schur, global_row, columns(), values());
    }
    schur->fillComplete();
    return schur;
}

/**
 * @brief Apply a block-triangular velocity-pressure preconditioner.
 *
 * Scratch MultiVectors are resized only when the input column count changes.
 * Packing and unpacking use the Tpetra device views. Like the underlying
 * Ifpack2 and MueLu objects, one instance must not be applied concurrently.
 *
 * @tparam Pack Tpetra type pack defining operator and multivector types.
 */
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
    using pressure_preconditioner_type =
        MueLu::TpetraOperator<
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
          d_momentum(std::move(momentum)),
          d_gradient(std::move(gradient)),
          d_schur(std::move(schur))
    {
        Teuchos::ParameterList momentum_parameters;
        momentum_parameters.set("relaxation: type", "Jacobi");
        momentum_parameters.set("relaxation: sweeps", 1);
        d_momentum_preconditioner =
            Ifpack2::Factory::create("RELAXATION", d_momentum);
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
        pressure_parameters.set("reuse: type", "full");
        pressure_parameters.sublist("coarse: params").set(
            "relaxation: type", "Jacobi");
        pressure_parameters.sublist("coarse: params").set(
            "relaxation: sweeps", 4);
        Teuchos::RCP<operator_type> pressure_operator = d_schur;
        d_pressure_preconditioner =
            MueLu::CreateTpetraPreconditioner(pressure_operator,
                                               pressure_parameters);
    }

    /**
     * @brief Test whether new numeric operators share the cached graphs.
     */
    bool is_compatible(
        const Teuchos::RCP<const matrix_type>& momentum,
        const std::array<Teuchos::RCP<matrix_type>, 3>& gradient,
        const Teuchos::RCP<matrix_type>& schur) const
    {
        if (momentum.is_null() || schur.is_null()
            || momentum.getRawPtr() != d_momentum.getRawPtr()
            || !same_graph(*d_momentum, *momentum)
            || !same_graph(*d_schur, *schur))
        {
            return false;
        }
        for (size_t component = 0; component < d_gradient.size(); ++component)
        {
            if (gradient[component].is_null()
                || !same_graph(*d_gradient[component],
                               *gradient[component]))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Refresh numeric preconditioner state while retaining setup.
     * @throws std::invalid_argument if an operator graph changed.
     */
    void update(
        Teuchos::RCP<const matrix_type> momentum,
        std::array<Teuchos::RCP<matrix_type>, 3> gradient,
        Teuchos::RCP<matrix_type> schur)
    {
        if (!is_compatible(momentum, gradient, schur))
        {
            throw std::invalid_argument(
                "CoupledSchurPreconditioner cannot reuse changed operator graphs.");
        }

        d_momentum_preconditioner->compute();
        MueLu::ReuseTpetraPreconditioner(
            schur, *d_pressure_preconditioner);
        d_momentum = std::move(momentum);
        d_gradient = std::move(gradient);
        d_schur = std::move(schur);
    }

    /** @brief Number of scratch-storage shapes allocated so far. */
    size_t scratch_allocations() const noexcept
    {
        return d_scratch_allocations;
    }

    Teuchos::RCP<const map_type> getDomainMap() const override
    {
        return d_coupled_map;
    }

    Teuchos::RCP<const map_type> getRangeMap() const override
    {
        return d_coupled_map;
    }

    /**
     * @brief Apply @f$Y \leftarrow \beta Y + \alpha M^{-1}X@f$.
     * @param mode Transpose mode; only `Teuchos::NO_TRANS` is supported.
     * @throws std::invalid_argument If a transpose application is requested.
     */
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
        ensure_scratch(num_vectors);
        auto& pressure_rhs = *d_pressure_rhs;
        auto& pressure_solution = *d_pressure_solution;
        auto& velocity_rhs = *d_velocity_rhs;
        auto& velocity_solution = *d_velocity_solution;
        auto& gradient_value = *d_gradient_value;
        auto& result = *d_result;

        const auto local_cells = d_cell_map->getLocalNumElements();
        const auto packed_entries = local_cells * num_vectors;
        using execution_space = typename Pack::execution_space;
        using range_policy = Kokkos::RangePolicy<
            execution_space, Kokkos::IndexType<size_t>>;
        {
            const auto input_view =
                input.getLocalViewDevice(Tpetra::Access::ReadOnly);
            const auto pressure_rhs_view =
                pressure_rhs.getLocalViewDevice(Tpetra::Access::OverwriteAll);
            const auto velocity_rhs_view =
                velocity_rhs.getLocalViewDevice(Tpetra::Access::OverwriteAll);
            Kokkos::parallel_for(
                "SimpleFluid::CoupledSchurPreconditioner::pack",
                range_policy(0, packed_entries),
                KOKKOS_LAMBDA(const size_t packed)
                {
                    const auto cell = packed / num_vectors;
                    const auto vector = packed % num_vectors;
                    pressure_rhs_view(cell, vector) =
                        input_view(4 * cell + 3, vector);
                    for (size_t component = 0; component < 3; ++component)
                    {
                        velocity_rhs_view(cell, 3 * vector + component) =
                            input_view(4 * cell + component, vector);
                    }
                });
        }

        d_pressure_preconditioner->apply(
            pressure_rhs, pressure_solution);

        for (size_t component = 0; component < d_gradient.size(); ++component)
        {
            gradient_value.putScalar(scalar_type{});
            d_gradient[component]->apply(
                pressure_solution, gradient_value);
            const auto gradient_view =
                gradient_value.getLocalViewDevice(Tpetra::Access::ReadOnly);
            const auto velocity_update_view =
                velocity_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite);
            Kokkos::parallel_for(
                "SimpleFluid::CoupledSchurPreconditioner::gradient_update",
                range_policy(0, packed_entries),
                KOKKOS_LAMBDA(const size_t packed)
                {
                    const auto cell = packed / num_vectors;
                    const auto vector = packed % num_vectors;
                    velocity_update_view(
                        cell, 3 * vector + component) -=
                        gradient_view(cell, vector);
                });
        }

        d_momentum_preconditioner->apply(
            velocity_rhs, velocity_solution);

        {
            const auto result_view =
                result.getLocalViewDevice(Tpetra::Access::OverwriteAll);
            const auto pressure_view =
                pressure_solution.getLocalViewDevice(Tpetra::Access::ReadOnly);
            const auto velocity_view =
                velocity_solution.getLocalViewDevice(Tpetra::Access::ReadOnly);
            Kokkos::parallel_for(
                "SimpleFluid::CoupledSchurPreconditioner::unpack",
                range_policy(0, packed_entries),
                KOKKOS_LAMBDA(const size_t packed)
                {
                    const auto cell = packed / num_vectors;
                    const auto vector = packed % num_vectors;
                    for (size_t component = 0; component < 3; ++component)
                    {
                        result_view(4 * cell + component, vector) =
                            velocity_view(cell, 3 * vector + component);
                    }
                    result_view(4 * cell + 3, vector) =
                        pressure_view(cell, vector);
                });
        }

        output.update(alpha, result, beta);
    }

private:
    static bool same_graph(
        const matrix_type& lhs,
        const matrix_type& rhs)
    {
        return lhs.isFillComplete()
            && rhs.isFillComplete()
            && lhs.getCrsGraph()->isIdenticalTo(*rhs.getCrsGraph());
    }

    void ensure_scratch(size_t num_vectors) const
    {
        if (num_vectors == d_scratch_num_vectors
            && !d_pressure_rhs.is_null())
        {
            return;
        }

        d_pressure_rhs = Teuchos::rcp(
            new multi_vector_type(d_cell_map, num_vectors, true));
        d_pressure_solution = Teuchos::rcp(
            new multi_vector_type(d_cell_map, num_vectors, true));
        d_velocity_rhs = Teuchos::rcp(
            new multi_vector_type(d_cell_map, 3 * num_vectors, true));
        d_velocity_solution = Teuchos::rcp(
            new multi_vector_type(d_cell_map, 3 * num_vectors, true));
        d_gradient_value = Teuchos::rcp(
            new multi_vector_type(d_cell_map, num_vectors, true));
        d_result = Teuchos::rcp(
            new multi_vector_type(d_coupled_map, num_vectors, true));
        d_scratch_num_vectors = num_vectors;
        ++d_scratch_allocations;
    }

    Teuchos::RCP<const map_type> d_coupled_map;
    Teuchos::RCP<const map_type> d_cell_map;
    Teuchos::RCP<const matrix_type> d_momentum;
    std::array<Teuchos::RCP<matrix_type>, 3> d_gradient;
    Teuchos::RCP<matrix_type> d_schur;
    Teuchos::RCP<momentum_preconditioner_type>
        d_momentum_preconditioner;
    Teuchos::RCP<pressure_preconditioner_type>
        d_pressure_preconditioner;
    mutable Teuchos::RCP<multi_vector_type> d_pressure_rhs;
    mutable Teuchos::RCP<multi_vector_type> d_pressure_solution;
    mutable Teuchos::RCP<multi_vector_type> d_velocity_rhs;
    mutable Teuchos::RCP<multi_vector_type> d_velocity_solution;
    mutable Teuchos::RCP<multi_vector_type> d_gradient_value;
    mutable Teuchos::RCP<multi_vector_type> d_result;
    mutable size_t d_scratch_num_vectors = 0;
    mutable size_t d_scratch_allocations = 0;
};

} // namespace detail

/**
 * @brief Assemble and solve monolithic cell velocity-pressure systems.
 *
 * The mesh is immutable for the solver lifetime. Cached mutable algebra makes
 * instances sequential-use objects; concurrent assembly or solve calls on the
 * same instance are unsupported.
 *
 * @tparam Pack Tpetra type pack defining distributed algebra types.
 */
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
    using graph_type = typename Pack::graph_type;
    using matrix_type = typename Pack::matrix_type;
    using vector_type = typename Pack::vector_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using operator_type = typename Pack::operator_type;
    using momentum_system_type = FVM::VectorTransportSystem<Pack>;
    using system_type = CoupledPressureVelocitySystem<Pack>;
    using result_type = CoupledPressureVelocityResult<scalar_type>;
    using preconditioner_type = detail::CoupledSchurPreconditioner<Pack>;
    using problem_type =
        Belos::LinearProblem<scalar_type, multi_vector_type, operator_type>;
    using solver_type =
        Belos::BlockGmresSolMgr<scalar_type, multi_vector_type, operator_type>;

    /**
     * @throws std::invalid_argument If @p mesh is null.
     */
    explicit CoupledPressureVelocitySolver(SP<const mesh_type> mesh)
        : d_mesh(EquationValidation::require_non_null_mesh(
              std::move(mesh), "CoupledPressureVelocitySolver")),
          d_coupled_map(detail::make_coupled_map(*d_mesh)),
          d_boundary_locations(
              FVM::detail::boundary_face_locations(*d_mesh))
    {
        d_cache_statistics.coupled_map_builds = 1;
    }

    /** @brief Select the coupled setup rebuild policy. */
    void set_rebuild_policy(CoupledRebuildPolicy policy)
    {
        if (d_rebuild_policy == policy)
        {
            return;
        }
        d_rebuild_policy = policy;
        clear_cache();
    }

    /** @brief Return the active coupled setup rebuild policy. */
    CoupledRebuildPolicy rebuild_policy() const noexcept
    {
        return d_rebuild_policy;
    }

    /** @brief Return setup-reuse instrumentation counters. */
    const CoupledPressureVelocityCacheStatistics&
    cache_statistics() const noexcept
    {
        return d_cache_statistics;
    }

    /**
     * @brief Discard cached assembly, preconditioner, and Krylov state.
     *
     * Instrumentation counters remain cumulative across this operation.
     */
    void clear_cache()
    {
        d_cached_system = {};
        d_static_geometry = {};
        d_schur_workspace.clear();
        d_gradient_stabilization_products = {};
        d_cached_momentum_graph = nullptr;
        d_cached_pressure_graph_signature.clear();
        d_preconditioner = Teuchos::null;
        d_belos_solver = Teuchos::null;
        d_last_belos_matrix = nullptr;
        d_solution = Teuchos::null;
    }

    /**
     * @brief Assemble an isothermal incompressible coupled system.
     * @param pressure Current physical pressure in Pa.
     * @param reference_density Positive density used to normalize pressure.
     */
    system_type assemble(
        const IncompressibleMomentumEquation<Pack>& momentum_equation,
        const velocity_field_type& velocity,
        const field_type& pressure,
        const face_flux_field_type& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options,
        scalar_type reference_density = scalar_type{1}) const
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
            time_options,
            reference_density);
    }

    /**
     * @brief Assemble a thermally buoyant coupled system.
     * @param pressure Current physical pressure in Pa.
     * @param reference_density Positive density used to normalize pressure.
     * @throws std::invalid_argument If supplied fields are incompatible.
     */
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
        bool density_feedback_enabled = false,
        const field_type* dynamic_viscosity_override = nullptr,
        const velocity_field_type* turbulent_kinetic_energy_gradient = nullptr,
        const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity = nullptr) const
    {
        EquationValidation::require_mesh_match(
            *d_mesh, velocity, "CoupledPressureVelocitySolver");
        EquationValidation::require_mesh_match(
            *d_mesh, pressure, "CoupledPressureVelocitySolver");
        EquationValidation::require_mesh_match(
            *d_mesh, temperature, "CoupledPressureVelocitySolver");
        if (dynamic_viscosity_override != nullptr && material == nullptr)
        {
            throw std::invalid_argument(
                "CoupledPressureVelocitySolver requires material fields "
                "when a dynamic-viscosity override is supplied.");
        }
        if (turbulent_kinetic_energy_gradient != nullptr)
        {
            EquationValidation::require_mesh_match(
                *d_mesh,
                *turbulent_kinetic_energy_gradient,
                "CoupledPressureVelocitySolver");
        }

        const auto* correction_field =
            time_options.non_orthogonal_treatment
                == FVM::NonOrthogonalTreatment::Implicit
          ? nullptr
          : &velocity;
        auto turbulence_source =
            [&](local_ordinal_type cell_lid)
                -> typename velocity_field_type::vec_type
        {
            return turbulent_kinetic_energy_gradient == nullptr
                ? typename velocity_field_type::vec_type{}
                : turbulent_kinetic_energy_gradient->value(cell_lid)
                    * scalar_type{-2.0 / 3.0};
        };
        typename BoussinesqMomentumEquation<Pack>::system_type momentum;
        if (material != nullptr)
        {
            momentum = momentum_equation.assemble_physical_system(
                velocity,
                face_fluxes,
                temperature,
                velocity_boundary_cache,
                time_options,
                *material,
                reference_density,
                density_feedback_enabled,
                turbulence_source,
                correction_field,
                dynamic_viscosity_override,
                boundary_dynamic_viscosity);
        }
        else if (turbulent_kinetic_energy_gradient != nullptr)
        {
            momentum = momentum_equation.assemble_system(
                velocity,
                face_fluxes,
                temperature,
                velocity_boundary_cache,
                time_options,
                turbulence_source,
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
            time_options,
            reference_density);
    }

private:
    using pressure_graph_signature_type =
        std::vector<std::pair<std::string, BoundaryConditionType>>;

    struct StaticGeometryCache
    {
        BoundaryConditionMap pressure_boundaries;
        scalar_type reference_density = {};
        std::vector<detail::AffinePressureGradientStencil<Pack>> stencils;
        std::array<Teuchos::RCP<matrix_type>, 3> gradient_operators;
        std::array<Teuchos::RCP<vector_type>, 3> gradient_constants;

        bool empty() const noexcept
        {
            return stencils.empty();
        }
    };

    static bool same_pressure_boundaries(
        const BoundaryConditionMap& lhs,
        const BoundaryConditionMap& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }
        for (const auto& [name, condition] : lhs)
        {
            const auto iter = rhs.find(name);
            if (iter == rhs.end()
                || iter->second.type != condition.type
                || iter->second.value != condition.value
                || iter->second.robin_coefficient
                   != condition.robin_coefficient)
            {
                return false;
            }
        }
        return true;
    }

    static pressure_graph_signature_type pressure_graph_signature(
        const BoundaryConditionMap& boundaries)
    {
        pressure_graph_signature_type signature;
        signature.reserve(boundaries.size());
        for (const auto& [name, condition] : boundaries)
        {
            signature.emplace_back(name, condition.type);
        }
        std::sort(
            signature.begin(), signature.end(),
            [](const auto& lhs, const auto& rhs)
            {
                return lhs.first < rhs.first;
            });
        return signature;
    }

    bool can_reuse_assembly_graph(
        const momentum_system_type& momentum,
        const pressure_graph_signature_type& pressure_signature) const
    {
        // BlockGmresSolMgr retains its last iteration object after
        // setProblem(null), and that object retains the previous coupled
        // operator until the next solve rebuilds the iteration.  Permit that
        // one dormant internal owner, but do not mutate a matrix still held
        // by a caller through a previously returned system.
        const auto dormant_belos_owns_cached_matrix =
            !d_belos_solver.is_null()
            && d_last_belos_matrix
                   == d_cached_system.matrix.getRawPtr();
        const auto internal_owner_limit =
            dormant_belos_owns_cached_matrix ? 2 : 1;
        return d_rebuild_policy
                   == CoupledRebuildPolicy::OnOperatorGraphChange
            && !d_cached_system.matrix.is_null()
            && d_cached_system.matrix.strong_count()
                   <= internal_owner_limit
            && momentum.matrix->getCrsGraph().getRawPtr()
               == d_cached_momentum_graph
            && pressure_signature == d_cached_pressure_graph_signature;
    }

    system_type assemble_coupled_system(
        const momentum_system_type& momentum,
        const velocity_field_type& velocity,
        const field_type& pressure,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options,
        scalar_type reference_density) const
    {
        if (!std::isfinite(reference_density)
            || reference_density <= scalar_type{})
        {
            throw std::invalid_argument(
                "CoupledPressureVelocitySolver requires a finite positive "
                "reference density.");
        }
        FVM::detail::validate_pressure_velocity_boundary_compatibility(
            velocity_boundary_cache,
            boundary_conditions.pressure);

        const auto pressure_signature =
            pressure_graph_signature(boundary_conditions.pressure);
        const auto reuse_assembly_graph =
            can_reuse_assembly_graph(momentum, pressure_signature);
        if (!reuse_assembly_graph)
        {
            d_schur_workspace.clear();
            d_gradient_stabilization_products = {};
        }

        const auto prepared_coupled =
            detail::prepare_coupled_matrix<Pack>(
                d_coupled_map, Teuchos::null,
                reuse_assembly_graph
                    ? d_cached_system.matrix
                    : Teuchos::null,
                128);
        const auto& coupled_matrix = prepared_coupled.matrix;
        auto coupled_rhs =
            reuse_assembly_graph && !d_cached_system.rhs.is_null()
          ? d_cached_system.rhs
          : Teuchos::rcp(new vector_type(d_coupled_map, true));
        coupled_rhs->putScalar(scalar_type{});

        std::array<detail::PreparedCoupledMatrix<Pack>, 3>
            prepared_gradient;
        std::array<detail::PreparedCoupledMatrix<Pack>, 3>
            prepared_divergence;
        for (size_t component = 0; component < 3; ++component)
        {
            prepared_gradient[component] =
                detail::prepare_coupled_matrix<Pack>(
                d_mesh->owned_cell_map(),
                d_mesh->overlap_cell_map(),
                reuse_assembly_graph
                    ? d_cached_system.gradient[component]
                    : Teuchos::null,
                16);
            prepared_divergence[component] =
                detail::prepare_coupled_matrix<Pack>(
                d_mesh->owned_cell_map(),
                d_mesh->overlap_cell_map(),
                reuse_assembly_graph
                    ? d_cached_system.divergence[component]
                    : Teuchos::null,
                16);
        }
        std::array<Teuchos::RCP<matrix_type>, 3> gradient;
        std::array<Teuchos::RCP<matrix_type>, 3> divergence;
        for (size_t component = 0; component < 3; ++component)
        {
            gradient[component] = prepared_gradient[component].matrix;
            divergence[component] = prepared_divergence[component].matrix;
        }
        auto prepared_pressure_stabilization =
            detail::prepare_coupled_matrix<Pack>(
                d_mesh->owned_cell_map(),
                Teuchos::null,
                reuse_assembly_graph
                    ? d_cached_system.pressure_stabilization
                    : Teuchos::null,
                32);
        auto& pressure_stabilization =
            prepared_pressure_stabilization.matrix;

        const auto reuse_static_geometry =
            d_rebuild_policy
                == CoupledRebuildPolicy::OnOperatorGraphChange
            && !d_static_geometry.empty()
            && d_static_geometry.reference_density == reference_density
            && same_pressure_boundaries(
                d_static_geometry.pressure_boundaries,
                boundary_conditions.pressure);
        if (reuse_static_geometry)
        {
            ++d_cache_statistics.static_geometry_reuses;
        }
        else
        {
            ++d_cache_statistics.static_geometry_builds;
            StaticGeometryCache rebuilt;
            rebuilt.pressure_boundaries = boundary_conditions.pressure;
            rebuilt.reference_density = reference_density;
            rebuilt.stencils =
                detail::pressure_gradient_stencils<Pack>(
                    *d_mesh,
                    boundary_conditions.pressure,
                    reference_density);
            for (size_t component = 0; component < 3; ++component)
            {
                rebuilt.gradient_operators[component] = Teuchos::rcp(
                    new matrix_type(
                        d_mesh->owned_cell_map(),
                        d_mesh->overlap_cell_map(),
                        16));
                rebuilt.gradient_constants[component] = Teuchos::rcp(
                    new vector_type(d_mesh->owned_cell_map(), true));
            }
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto& stencil = rebuilt.stencils[owned];
                for (size_t component = 0; component < 3; ++component)
                {
                    Teuchos::Array<local_ordinal_type> columns;
                    Teuchos::Array<scalar_type> values;
                    columns.reserve(stencil.entries.size());
                    values.reserve(stencil.entries.size());
                    for (const auto& entry : stencil.entries)
                    {
                        columns.push_back(entry.cell_lid);
                        values.push_back(
                            entry.coefficient.component(component));
                    }
                    rebuilt.gradient_operators[component]
                        ->insertLocalValues(
                            cell_lid, columns(), values());
                    rebuilt.gradient_constants[component]
                        ->replaceLocalValue(
                            cell_lid,
                            stencil.constant.component(component));
                }
            }
            for (auto& pressure_gradient_operator :
                 rebuilt.gradient_operators)
            {
                pressure_gradient_operator->fillComplete();
            }
            d_static_geometry = std::move(rebuilt);
        }
        const auto& pressure_gradient_operators =
            d_static_geometry.gradient_operators;
        const auto& pressure_gradient_constants =
            d_static_geometry.gradient_constants;
        const auto& boundary_locations = d_boundary_locations;
        const auto momentum_rhs = momentum.rhs->getLocalViewHost(
            Tpetra::Access::ReadOnly);
        const auto momentum_col_map = momentum.matrix->getColMap();

        // A physical Dirichlet face fixes the pressure level on every rank.
        // Otherwise one global cell retains the all-Neumann gauge.
        int local_has_dirichlet_pressure = 0;
        for (const auto& [batch_id, boundary_batch] :
             d_mesh->boundary_batches())
        {
            const auto condition_iter =
                boundary_conditions.pressure.find(
                    d_mesh->boundary_batch_name(batch_id));
            if (condition_iter == boundary_conditions.pressure.end()
                || condition_iter->second.type
                   != BoundaryConditionType::Dirichlet)
            {
                continue;
            }
            for (const auto face_lid : boundary_batch.face_lids)
            {
                if (d_mesh->is_owned_face(face_lid)
                    && d_mesh->is_boundary_face(face_lid))
                {
                    local_has_dirichlet_pressure = 1;
                    break;
                }
            }
            if (local_has_dirichlet_pressure != 0)
            {
                break;
            }
        }
        int global_has_dirichlet_pressure = 0;
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(),
            Teuchos::REDUCE_MAX,
            1,
            &local_has_dirichlet_pressure,
            &global_has_dirichlet_pressure);
        const std::optional<global_ordinal_type> pressure_gauge_gid =
            global_has_dirichlet_pressure != 0
          ? std::optional<global_ordinal_type>{}
          : std::optional<global_ordinal_type>{
                d_mesh->owned_cell_map()->getMinAllGlobalIndex()};

        std::vector<scalar_type> continuity_rhs_values(
            d_mesh->num_owned_cells(), scalar_type{});

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
                const auto normalized_pressure_value =
                    pressure_condition.value / reference_density;

                if (pressure_condition.type
                    == BoundaryConditionType::Dirichlet)
                {
                    // Pressure outlets use the same boundary flux as the
                    // Rhie-Chow reconstruction:
                    // u_P.A + dt*k*q_P + dt*grad(q)_P.A - dt*k*q_D.
                    for (size_t component = 0; component < 3; ++component)
                    {
                        momentum_boundary_rhs[component] -=
                            normalized_pressure_value
                            * area_components[component];
                        detail::add_entry(
                            divergence_rows[component], cell_lid,
                            area_components[component]);
                    }

                    const auto boundary_coefficient =
                        FVM::detail::boundary_diffusion_coefficient(
                            *d_mesh,
                            face_lid,
                            cell_lid,
                            scalar_type{1});
                    detail::add_entry(
                        stabilization_row,
                        cell_lid,
                        time_options.time_step
                            * boundary_coefficient);
                    continuity_rhs +=
                        time_options.time_step
                      * boundary_coefficient
                      * normalized_pressure_value;
                }
                else
                {
                    for (size_t component = 0; component < 3; ++component)
                    {
                        detail::add_entry(
                            gradient_rows[component], cell_lid,
                            area_components[component]);
                        momentum_boundary_rhs[component] -=
                            normalized_pressure_value
                            * FVM::detail::boundary_normal_distance(
                                *d_mesh, face_lid, cell_lid)
                            * area_components[component];
                    }
                }

                if (pressure_condition.type
                    != BoundaryConditionType::Dirichlet)
                {
                    const auto velocity_type =
                        velocity_boundary_cache.type.at(
                            location.batch_id);
                    const auto prescribed =
                        velocity_type == BoundaryConditionType::Slip
                      ? FVM::detail::slip_face_velocity(
                            velocity, face_lid)
                      : velocity_boundary_cache.value.at(
                            location.batch_id)[location.in_batch_id];
                    continuity_rhs -= prescribed.dot(area);
                }
            }
            continuity_rhs_values[owned] = continuity_rhs;

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
                detail::add_coupled_local_values<Pack>(
                    prepared_gradient[component],
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
                detail::add_coupled_local_values<Pack>(
                    prepared_divergence[component],
                    cell_lid, columns(), values());
            }

            {
                Teuchos::Array<global_ordinal_type> columns;
                Teuchos::Array<scalar_type> values;
                columns.reserve(stabilization_row.size());
                values.reserve(stabilization_row.size());
                for (const auto& [column, value] : stabilization_row)
                {
                    columns.push_back(
                        d_mesh->overlap_cell_map()->getGlobalElement(
                            column));
                    values.push_back(value);
                }
                detail::add_coupled_global_values<Pack>(
                    prepared_pressure_stabilization,
                    cell_gid, columns(), values());
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
                detail::add_coupled_global_values<Pack>(
                    prepared_coupled,
                    coupled_row, columns(), values());
                coupled_rhs->replaceGlobalValue(
                    coupled_row,
                    momentum_rhs(cell_lid, component)
                    + momentum_boundary_rhs[component]);
            }
        }

        for (size_t component = 0; component < 3; ++component)
        {
            gradient[component]->fillComplete();
            divergence[component]->fillComplete();
        }

        // The divergence matrices contain the face interpolation weights.
        // D*G therefore assembles the same owner-neighbor gradient average
        // used by pressure_weighted_face_fluxes, including partition faces.
        for (size_t component = 0; component < 3; ++component)
        {
            vector_type constant_divergence(
                d_mesh->owned_cell_map(), true);
            divergence[component]->apply(
                *pressure_gradient_constants[component],
                constant_divergence);
            const auto constant_divergence_data =
                constant_divergence.getData();
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                continuity_rhs_values[owned] -=
                    time_options.time_step
                  * constant_divergence_data[owned];
            }

            auto& gradient_stabilization =
                d_gradient_stabilization_products[component];
            if (gradient_stabilization.is_null())
            {
                gradient_stabilization = Teuchos::rcp(
                    new matrix_type(
                        divergence[component]->getRowMap(), 32));
            }
            Tpetra::MatrixMatrix::Multiply(
                *divergence[component], false,
                *pressure_gradient_operators[component], false,
                *gradient_stabilization, true);
            const auto product_col_map =
                gradient_stabilization->getColMap();
            for (size_t row = 0;
                 row < gradient_stabilization->getLocalNumRows();
                 ++row)
            {
                typename matrix_type::local_inds_host_view_type
                    product_columns;
                typename matrix_type::values_host_view_type product_values;
                gradient_stabilization->getLocalRowView(
                    static_cast<local_ordinal_type>(row),
                    product_columns, product_values);
                Teuchos::Array<global_ordinal_type> columns(
                    product_columns.extent(0));
                Teuchos::Array<scalar_type> values(
                    product_values.extent(0));
                for (size_t entry = 0;
                     entry < product_columns.extent(0);
                     ++entry)
                {
                    const auto column_gid =
                        product_col_map->getGlobalElement(
                            product_columns[entry]);
                    columns[entry] = column_gid;
                    values[entry] =
                        time_options.time_step * product_values[entry];
                }
                const auto row_gid =
                    gradient_stabilization->getRowMap()->getGlobalElement(
                        static_cast<local_ordinal_type>(row));
                detail::add_coupled_global_values<Pack>(
                    prepared_pressure_stabilization,
                    row_gid,
                    columns(), values());
            }
        }
        pressure_stabilization->fillComplete();

        const auto stabilization_col_map =
            pressure_stabilization->getColMap();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto cell_gid =
                d_mesh->owned_cell_map()->getGlobalElement(cell_lid);
            const auto pressure_row = 4 * cell_gid + 3;
            if (pressure_gauge_gid
                && cell_gid == *pressure_gauge_gid)
            {
                Teuchos::Array<global_ordinal_type> columns{pressure_row};
                Teuchos::Array<scalar_type> values{scalar_type{1}};
                detail::add_coupled_global_values<Pack>(
                    prepared_coupled,
                    pressure_row, columns(), values());
                coupled_rhs->replaceGlobalValue(
                    pressure_row, scalar_type{});
                continue;
            }

            Teuchos::Array<global_ordinal_type> columns;
            Teuchos::Array<scalar_type> values;
            for (size_t component = 0; component < 3; ++component)
            {
                typename matrix_type::local_inds_host_view_type
                    local_columns;
                typename matrix_type::values_host_view_type local_values;
                divergence[component]->getLocalRowView(
                    cell_lid, local_columns, local_values);
                const auto divergence_col_map =
                    divergence[component]->getColMap();
                for (size_t entry = 0;
                     entry < local_columns.extent(0);
                     ++entry)
                {
                    const auto column_cell_gid =
                        divergence_col_map->getGlobalElement(
                            local_columns[entry]);
                    columns.push_back(
                        4 * column_cell_gid
                      + static_cast<global_ordinal_type>(component));
                    values.push_back(local_values[entry]);
                }
            }

            typename matrix_type::local_inds_host_view_type
                local_columns;
            typename matrix_type::values_host_view_type local_values;
            pressure_stabilization->getLocalRowView(
                cell_lid, local_columns, local_values);
            for (size_t entry = 0;
                 entry < local_columns.extent(0);
                 ++entry)
            {
                const auto column_cell_gid =
                    stabilization_col_map->getGlobalElement(
                        local_columns[entry]);
                columns.push_back(4 * column_cell_gid + 3);
                values.push_back(local_values[entry]);
            }
            detail::add_coupled_global_values<Pack>(
                prepared_coupled,
                pressure_row, columns(), values());
            coupled_rhs->replaceGlobalValue(
                pressure_row, continuity_rhs_values[owned]);
        }

        coupled_matrix->fillComplete(d_coupled_map, d_coupled_map);
        const auto coupled_overlap_map = coupled_matrix->getColMap();

        bool reused_schur_products = false;
        auto schur = detail::build_schur_approximation<Pack>(
            *momentum.matrix, gradient, divergence,
            *pressure_stabilization, pressure_gauge_gid,
            reuse_assembly_graph ? d_cached_system.schur : Teuchos::null,
            &d_schur_workspace,
            &reused_schur_products);
        if (reuse_assembly_graph)
        {
            ++d_cache_statistics.matrix_graph_reuses;
        }
        if (reused_schur_products)
        {
            ++d_cache_statistics.schur_product_reuses;
        }

        system_type assembled{
            d_coupled_map,
            coupled_overlap_map,
            coupled_matrix,
            coupled_rhs,
            momentum.matrix,
            std::move(gradient),
            std::move(divergence),
            std::move(pressure_stabilization),
            std::move(schur),
            reference_density};
        d_cached_momentum_graph =
            momentum.matrix->getCrsGraph().getRawPtr();
        d_cached_pressure_graph_signature = pressure_signature;
        d_cached_system = assembled;
        return assembled;
    }

public:
    /**
     * @brief Solve a coupled system and update velocity and physical pressure.
     * @param velocity Velocity field updated on convergence or termination.
     * @param pressure Physical pressure field updated in Pa.
     * @return Convergence status and Krylov statistics.
     * @throws std::invalid_argument If the pressure normalization is invalid.
     * @throws std::runtime_error If the Belos problem cannot be initialized.
     */
    result_type solve(
        const system_type& system,
        velocity_field_type& velocity,
        field_type& pressure,
        const LinearSolverOptions& options) const
    {
        if (!std::isfinite(system.reference_density)
            || system.reference_density <= scalar_type{})
        {
            throw std::invalid_argument(
                "Coupled pressure-velocity system requires a finite positive "
                "reference density.");
        }

        if (d_solution.is_null()
            || !d_solution->getMap()->isSameAs(*system.map)
            || d_rebuild_policy == CoupledRebuildPolicy::Always)
        {
            d_solution = Teuchos::rcp(
                new vector_type(system.map, true));
        }
        auto solution = d_solution;
        solution->putScalar(scalar_type{});
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
                4 * cell_gid + 3,
                pressure.value(cell_lid) / system.reference_density);
        }

        const auto reuse_preconditioner =
            d_rebuild_policy
                == CoupledRebuildPolicy::OnOperatorGraphChange
            && !d_preconditioner.is_null()
            && d_preconditioner->is_compatible(
                system.momentum, system.gradient, system.schur);
        if (reuse_preconditioner)
        {
            d_preconditioner->update(
                system.momentum, system.gradient, system.schur);
            ++d_cache_statistics.preconditioner_numeric_reuses;
        }
        else
        {
            d_preconditioner = Teuchos::rcp(
                new preconditioner_type(
                system.map,
                d_mesh->owned_cell_map(),
                system.momentum,
                system.gradient,
                system.schur));
            ++d_cache_statistics.preconditioner_builds;
        }
        const auto scratch_allocations_before =
            d_preconditioner->scratch_allocations();
        Teuchos::RCP<const operator_type> matrix = system.matrix;
        Teuchos::RCP<const operator_type> right_preconditioner =
            d_preconditioner;
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
        parameters->set(
            "Implicit Residual Scaling", "Norm of RHS");
        parameters->set(
            "Explicit Residual Scaling", "Norm of RHS");
        parameters->set("Flexible Gmres", true);
        parameters->set("Block Size", 4);
        parameters->set(
            "Num Blocks",
            std::max(
                1,
                std::min(
                    std::max(1, options.max_iterations / 4),
                    static_cast<int>(
                        system.map->getGlobalNumElements() / 4))));
        if (d_belos_solver.is_null()
            || d_rebuild_policy == CoupledRebuildPolicy::Always)
        {
            d_belos_solver = Teuchos::rcp(
                new solver_type(problem, parameters));
            ++d_cache_statistics.belos_solver_builds;
        }
        else
        {
            d_belos_solver->setProblem(problem);
            d_belos_solver->setParameters(parameters);
            ++d_cache_statistics.belos_solver_reuses;
        }
        const auto converged =
            d_belos_solver->solve() == Belos::Converged;
        d_last_belos_matrix = system.matrix.getRawPtr();
        const auto iterations = d_belos_solver->getNumIters();
        const auto achieved_tolerance = d_belos_solver->achievedTol();
        d_cache_statistics.preconditioner_scratch_allocations +=
            d_preconditioner->scratch_allocations()
          - scratch_allocations_before;

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
                cell_lid,
                system.reference_density
                    * solution_view(4 * owned + 3, 0));
        }
        d_mesh->sync_periodic_boundaries(velocity);
        d_mesh->sync_periodic_boundaries(pressure);
        d_belos_solver->setProblem(Teuchos::null);

        return {
            converged,
            iterations,
            achieved_tolerance};
    }

private:
    SP<const mesh_type> d_mesh;
    CoupledRebuildPolicy d_rebuild_policy =
        CoupledRebuildPolicy::OnOperatorGraphChange;
    Teuchos::RCP<const typename Pack::map_type> d_coupled_map;
    std::vector<FVM::detail::BoundaryFaceLocation<mesh_type>>
        d_boundary_locations;
    mutable CoupledPressureVelocityCacheStatistics d_cache_statistics;
    mutable system_type d_cached_system;
    mutable StaticGeometryCache d_static_geometry;
    mutable detail::CoupledSchurWorkspace<Pack> d_schur_workspace;
    mutable std::array<Teuchos::RCP<matrix_type>, 3>
        d_gradient_stabilization_products;
    mutable const graph_type* d_cached_momentum_graph = nullptr;
    mutable pressure_graph_signature_type
        d_cached_pressure_graph_signature;
    mutable Teuchos::RCP<preconditioner_type> d_preconditioner;
    mutable Teuchos::RCP<solver_type> d_belos_solver;
    mutable const matrix_type* d_last_belos_matrix = nullptr;
    mutable Teuchos::RCP<vector_type> d_solution;
};

} // namespace SimpleFluid
