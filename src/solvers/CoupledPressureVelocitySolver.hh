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

#include "SimpleFluidExport.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/OperatorDetails.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/EquationForward.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "solvers/BelosLinearSolver.hh"

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

template<TpetraTypePack Pack> struct MaterialPropertyFields;

struct TimeStepperOptions;

namespace FVM
{
template<TpetraTypePack Pack> struct VectorTransportSystem;

template<TpetraTypePack Pack> struct VelocityBoundaryCache;
} // namespace FVM

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
template<TpetraTypePack Pack> struct CoupledPressureVelocitySystem
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
template<class Scalar> struct CoupledPressureVelocityResult
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
    Always = 1                 ///< Rebuild operator-dependent setup each call.
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
SIMPLEFLUID_SOLVERS_LOCAL
void add_entry(std::unordered_map<Column, Scalar>& row, Column column, Scalar value);

/**
 * @brief Build the four-unknown-per-cell map for a coupled system.
 *
 * @tparam Pack Tpetra type pack defining the distributed map.
 * @param mesh Mesh that owns the cell distribution.
 * @return Coupled velocity-pressure map.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL
auto make_coupled_map(const Mesh<Pack>& mesh) -> Teuchos::RCP<const typename Pack::map_type>;

/** @brief Matrix reset for cached-graph assembly. */
template<TpetraTypePack Pack> struct PreparedCoupledMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
};

/**
 * @brief Allocate a matrix or reset a compatible cached graph.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL
PreparedCoupledMatrix<Pack> prepare_coupled_matrix(const Teuchos::RCP<const typename Pack::map_type>& row_map,
    const Teuchos::RCP<const typename Pack::map_type>& column_map,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row);

/** @brief Insert into a fresh local graph or sum into a reused one. */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL
void add_coupled_local_values(const PreparedCoupledMatrix<Pack>& prepared, typename Pack::local_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::local_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values);

/** @brief Insert into a fresh global graph or sum into a reused one. */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL
void add_coupled_global_values(const PreparedCoupledMatrix<Pack>& prepared, typename Pack::global_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::global_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values);

/**
 * @brief Affine least-squares stencil for a normalized pressure gradient.
 *
 * @tparam Pack Tpetra type pack defining mesh ordinals and vectors.
 */
template<TpetraTypePack Pack> struct AffinePressureGradientStencil
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
SIMPLEFLUID_SOLVERS_LOCAL auto pressure_gradient_stencils(
    const Mesh<Pack>& mesh,
    const BoundaryConditionMap& boundary_conditions,
    typename Pack::scalar_type reference_density)
    -> std::vector<AffinePressureGradientStencil<Pack>>;

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
SIMPLEFLUID_SOLVERS_LOCAL
Teuchos::RCP<typename Pack::matrix_type> scaled_gradient_matrix(const typename Pack::matrix_type& gradient,
    const typename Pack::vector_type& inverse_diagonal,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/** @brief Persistent scratch matrices for Schur numeric updates. */
template<TpetraTypePack Pack> struct CoupledSchurWorkspace
{
    Teuchos::RCP<typename Pack::vector_type> inverse_diagonal;
    std::array<Teuchos::RCP<typename Pack::matrix_type>, 3> scaled_gradient;
    std::array<Teuchos::RCP<typename Pack::matrix_type>, 3> product;

    SIMPLEFLUID_SOLVERS_LOCAL
    void clear();
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
SIMPLEFLUID_SOLVERS_LOCAL
Teuchos::RCP<typename Pack::matrix_type> build_schur_approximation(const typename Pack::matrix_type& momentum,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& gradient,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& divergence,
    const typename Pack::matrix_type& pressure_stabilization,
    std::optional<typename Pack::global_ordinal_type> pressure_gauge_gid,
    Teuchos::RCP<typename Pack::matrix_type> cached_schur = Teuchos::null,
    CoupledSchurWorkspace<Pack>* workspace = nullptr, bool* reused_products = nullptr);

/**
 * @brief Apply a block-triangular velocity-pressure preconditioner.
 *
 * Scratch MultiVectors are resized only when the input column count changes.
 * Packing and unpacking use the Tpetra device views. Like the underlying
 * Ifpack2 and MueLu objects, one instance must not be applied concurrently.
 *
 * @tparam Pack Tpetra type pack defining operator and multivector types.
 */
template<TpetraTypePack Pack> class CoupledSchurPreconditioner;

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
class SIMPLEFLUID_SOLVERS_EXPORT CoupledPressureVelocitySolver
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
    using problem_type = Belos::LinearProblem<scalar_type, multi_vector_type, operator_type>;
    using solver_type = Belos::BlockGmresSolMgr<scalar_type, multi_vector_type, operator_type>;

    /**
     * @throws std::invalid_argument If @p mesh is null.
     */
    explicit CoupledPressureVelocitySolver(SP<const mesh_type> mesh);
    ~CoupledPressureVelocitySolver();

    /** @brief Select the coupled setup rebuild policy. */
    void set_rebuild_policy(CoupledRebuildPolicy policy);

    /** @brief Return the active coupled setup rebuild policy. */
    CoupledRebuildPolicy rebuild_policy() const noexcept;

    /** @brief Return setup-reuse instrumentation counters. */
    const CoupledPressureVelocityCacheStatistics& cache_statistics() const noexcept;

    /**
     * @brief Discard cached assembly, preconditioner, and Krylov state.
     *
     * Instrumentation counters remain cumulative across this operation.
     */
    void clear_cache();

    /**
     * @brief Assemble an isothermal incompressible coupled system.
     * @param pressure Current physical pressure in Pa.
     * @param reference_density Positive density used to normalize pressure.
     */
    system_type assemble(const IncompressibleMomentumEquation<Pack>& momentum_equation,
        const velocity_field_type& velocity, const field_type& pressure, const face_flux_field_type& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        scalar_type reference_density = scalar_type{1}) const;

    /**
     * @brief Assemble a thermally buoyant coupled system.
     * @param pressure Current physical pressure in Pa.
     * @param reference_density Positive density used to normalize pressure.
     * @throws std::invalid_argument If supplied fields are incompatible.
     */
    system_type assemble(const BoussinesqMomentumEquation<Pack>& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const field_type& temperature, const face_flux_field_type& face_fluxes,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        const MaterialPropertyFields<Pack>* material = nullptr, scalar_type reference_density = scalar_type{1},
        bool density_feedback_enabled = false, const field_type* dynamic_viscosity_override = nullptr,
        const velocity_field_type* turbulent_kinetic_energy_gradient = nullptr,
        const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity = nullptr) const;

private:
    using pressure_graph_signature_type = std::vector<std::pair<std::string, BoundaryConditionType>>;

    struct StaticGeometryCache
    {
        BoundaryConditionMap pressure_boundaries;
        scalar_type reference_density = {};
        std::vector<detail::AffinePressureGradientStencil<Pack>> stencils;
        std::array<Teuchos::RCP<matrix_type>, 3> gradient_operators;
        std::array<Teuchos::RCP<vector_type>, 3> gradient_constants;

        SIMPLEFLUID_SOLVERS_LOCAL
        bool empty() const noexcept;
    };

    SIMPLEFLUID_SOLVERS_LOCAL
    static bool same_pressure_boundaries(const BoundaryConditionMap& lhs, const BoundaryConditionMap& rhs);

    SIMPLEFLUID_SOLVERS_LOCAL
    static pressure_graph_signature_type pressure_graph_signature(const BoundaryConditionMap& boundaries);

    SIMPLEFLUID_SOLVERS_LOCAL
    bool can_reuse_assembly_graph(
        const momentum_system_type& momentum, const pressure_graph_signature_type& pressure_signature) const;

    SIMPLEFLUID_SOLVERS_LOCAL
    system_type assemble_coupled_system(const momentum_system_type& momentum, const velocity_field_type& velocity,
        const field_type& pressure, const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        scalar_type reference_density) const;

public:
    /**
     * @brief Solve a coupled system and update velocity and physical pressure.
     * @param velocity Velocity field updated on convergence or termination.
     * @param pressure Physical pressure field updated in Pa.
     * @return Convergence status and Krylov statistics.
     * @throws std::invalid_argument If the pressure normalization is invalid.
     * @throws std::runtime_error If the Belos problem cannot be initialized.
     */
    result_type solve(const system_type& system, velocity_field_type& velocity, field_type& pressure,
        const LinearSolverOptions& options) const;

private:
    SP<const mesh_type> d_mesh;
    CoupledRebuildPolicy d_rebuild_policy = CoupledRebuildPolicy::OnOperatorGraphChange;
    Teuchos::RCP<const typename Pack::map_type> d_coupled_map;
    std::vector<FVM::detail::BoundaryFaceLocation<mesh_type>> d_boundary_locations;
    mutable CoupledPressureVelocityCacheStatistics d_cache_statistics;
    mutable system_type d_cached_system;
    mutable StaticGeometryCache d_static_geometry;
    mutable detail::CoupledSchurWorkspace<Pack> d_schur_workspace;
    mutable std::array<Teuchos::RCP<matrix_type>, 3> d_gradient_stabilization_products;
    mutable const graph_type* d_cached_momentum_graph = nullptr;
    mutable pressure_graph_signature_type d_cached_pressure_graph_signature;
    mutable Teuchos::RCP<preconditioner_type> d_preconditioner;
    mutable Teuchos::RCP<solver_type> d_belos_solver;
    mutable const matrix_type* d_last_belos_matrix = nullptr;
    mutable Teuchos::RCP<vector_type> d_solution;
};

/**
 * @brief Library-provided coupled solver specialization.
 *
 * Additional Tpetra packs require a deliberate explicit instantiation in the
 * solver implementation target.
 */
extern template class CoupledPressureVelocitySolver<DefaultTpetraTypes>;

} // namespace SimpleFluid
