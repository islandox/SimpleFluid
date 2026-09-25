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

#include "FVM/BoundaryCache.hh"
#include "FVM/CellGradientScheme.hh"
#include "FVM/details/OperatorDetails.hh"
#include "SimpleFluidExport.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/CoupledOperatorBackend.hh"
#include "equations/EquationForward.hh"
#include "equations/VolumeContinuityTarget.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/MeshFieldTraits.hh"
#include "fields/VectorCellField.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/CoupledBlockOperator.hh"

#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

template<TpetraTypePack Pack, class MeshType> struct MaterialPropertyFields;

struct TimeStepperOptions;

namespace FVM
{
struct ALEControlVolumeState;

template<TpetraTypePack Pack> struct VectorTransportSystem;

template<TpetraTypePack Pack> struct VelocityBoundaryCache;

template<TpetraTypePack Pack, class MeshType> struct FieldStoredVelocityBoundaryCache;
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
    using global_ordinal_type = typename Pack::global_ordinal_type;

    Teuchos::RCP<const map_type> map;
    Teuchos::RCP<const map_type> overlap_map;
    Teuchos::RCP<matrix_type> matrix;
    Teuchos::RCP<vector_type> rhs;
    Teuchos::RCP<matrix_type> momentum;
    std::array<Teuchos::RCP<matrix_type>, 3> gradient;
    std::array<Teuchos::RCP<matrix_type>, 3> divergence;
    Teuchos::RCP<matrix_type> pressure_stabilization;
    /// Null when assembled for residual evaluation without linear-solve setup.
    Teuchos::RCP<matrix_type> schur;
    scalar_type reference_density = scalar_type{1};
    std::optional<global_ordinal_type> pressure_gauge_gid;
    std::uint64_t geometry_epoch = 0;
    std::uint64_t continuity_target_generation = 0;
    std::uint64_t fixed_boundary_flux_revision = 0;
    /// True operator. matrix and overlap_map are null for block_composite;
    /// communication uses the scalar block maps/importers instead.
    Teuchos::RCP<const typename Pack::operator_type> linear_operator;
    std::shared_ptr<FVM::NumericAssemblyLease> numeric_lease;
    CoupledOperatorBackend operator_backend = CoupledOperatorBackend::Assembled;
    CoupledWorkspacePolicy workspace_policy = CoupledWorkspacePolicy::CachedProducts;
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
    VolumeContinuityResiduals<Scalar> continuity;
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

/** Select whether assembly also prepares the Schur approximation for a solve.
 * ResidualOnly preserves the complete affine operator and RHS, including
 * pressure stabilization, boundary terms, continuity targets, and the gauge.
 */
enum class CoupledAssemblyPurpose : std::uint8_t
{
    LinearSolve = 0,
    ResidualOnly = 1
};

/**
 * @brief Observable counters for coupled setup reuse.
 *
 * The counters are intended for regression tests and lightweight runtime
 * instrumentation. They do not participate in the numerical algorithm.
 */
struct CoupledPressureVelocityCacheStatistics
{
    size_t cell_face_cache_builds = 0;
    size_t coupled_matrix_builds = 0;
    size_t composite_operator_builds = 0;
    size_t streamed_product_peak = 0;
    size_t streamed_products_live = 0;
    size_t coupled_map_builds = 0;
    size_t static_geometry_builds = 0;
    size_t static_geometry_reuses = 0;
    size_t matrix_graph_reuses = 0;
    size_t geometry_block_reuses = 0;
    size_t schur_slot_builds = 0;
    size_t schur_slot_reuses = 0;
    /// Completed Schur numeric assemblies, including compatible graph reuse.
    size_t schur_builds = 0;
    /// Computed D_i diag(A_m)^-1 G_i products, including numeric refreshes.
    size_t schur_product_builds = 0;
    size_t schur_product_reuses = 0;
    size_t preconditioner_builds = 0;
    size_t preconditioner_numeric_reuses = 0;
    size_t belos_solver_builds = 0;
    size_t belos_solver_reuses = 0;
    size_t preconditioner_scratch_allocations = 0;
};

/** Locally observed live CRS device-view payload, deduplicated by allocation address.
 * Excludes maps/imports, allocator metadata, Ifpack2/MueLu internals and Krylov
 * storage. On host-only Kokkos these are host allocations, counted once.
 */
struct CoupledStorageStatistics
{
    size_t live_matrices = 0;
    size_t graph_bytes = 0;
    size_t value_bytes = 0;
    size_t composite_scratch_bytes = 0;
    size_t composite_scratch_allocations = 0;
};

namespace detail
{

/**
 * @brief Build the four-unknown-per-cell map for a coupled system.
 *
 * @tparam Pack Tpetra type pack defining the distributed map.
 * @param mesh Mesh that owns the cell distribution.
 * @return Coupled velocity-pressure map.
 */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_SOLVERS_LOCAL auto make_coupled_map(const MeshType& mesh) -> Teuchos::RCP<const typename Pack::map_type>;

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
SIMPLEFLUID_SOLVERS_LOCAL PreparedCoupledMatrix<Pack> prepare_coupled_matrix(
    const Teuchos::RCP<const typename Pack::map_type>& row_map,
    const Teuchos::RCP<const typename Pack::map_type>& column_map,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row);

/** @brief Insert into a fresh local graph or sum into a reused one. */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL void add_coupled_local_values(const PreparedCoupledMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row, const Teuchos::ArrayView<const typename Pack::local_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values);

/** @brief Insert into a fresh global graph or sum into a reused one. */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL void add_coupled_global_values(const PreparedCoupledMatrix<Pack>& prepared,
    typename Pack::global_ordinal_type row, const Teuchos::ArrayView<const typename Pack::global_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values);

/**
 * @brief Affine least-squares stencil for a normalized pressure gradient.
 *
 * @tparam Pack Tpetra type pack defining mesh ordinals and vectors.
 */
template<TpetraTypePack Pack, class MeshType> struct AffinePressureGradientStencil
{
    FVM::detail::LeastSquaresGradientStencil<MeshType> entries;
    typename MeshType::Vec3 constant{};
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
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_SOLVERS_LOCAL auto pressure_gradient_stencils(
    const MeshType& mesh, const BoundaryConditionMap& boundary_conditions, typename Pack::scalar_type reference_density)
    -> std::vector<AffinePressureGradientStencil<Pack, MeshType>>;

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
SIMPLEFLUID_SOLVERS_LOCAL Teuchos::RCP<typename Pack::matrix_type> scaled_gradient_matrix(
    const typename Pack::matrix_type& gradient, const typename Pack::vector_type& inverse_diagonal,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/** @brief Persistent scratch matrices for Schur numeric updates. */
template<TpetraTypePack Pack> struct CoupledSchurWorkspace
{
    Teuchos::RCP<typename Pack::vector_type> inverse_diagonal;
    std::array<Teuchos::RCP<typename Pack::matrix_type>, 3> scaled_gradient;
    std::array<Teuchos::RCP<typename Pack::matrix_type>, 3> product;
    // Source entry order maps directly into the completed Schur CRS values.
    // Graph owners keep every recorded slot tied to its symbolic generation.
    std::array<Teuchos::RCP<const typename Pack::matrix_type::crs_graph_type>, 4> source_graphs;
    Teuchos::RCP<const typename Pack::matrix_type::crs_graph_type> schur_graph;
    std::array<std::vector<size_t>, 4> contribution_slots;
    std::vector<size_t> diagonal_slots;
    std::optional<typename Pack::global_ordinal_type> pressure_gauge_gid;

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
 * @param policy Selects whether sparse products are cached or streamed.
 * @param statistics Optional counters for product construction and reuse.
 * @return Assembled Schur-complement approximation.
 * @throws std::runtime_error if the momentum diagonal is singular.
 */
template<TpetraTypePack Pack>
SIMPLEFLUID_SOLVERS_LOCAL Teuchos::RCP<typename Pack::matrix_type> build_schur_approximation(
    const typename Pack::matrix_type& momentum, const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& gradient,
    const std::array<Teuchos::RCP<typename Pack::matrix_type>, 3>& divergence,
    const typename Pack::matrix_type& pressure_stabilization,
    std::optional<typename Pack::global_ordinal_type> pressure_gauge_gid,
    Teuchos::RCP<typename Pack::matrix_type> cached_schur = Teuchos::null,
    CoupledSchurWorkspace<Pack>* workspace = nullptr, bool* reused_products = nullptr,
    CoupledWorkspacePolicy policy = CoupledWorkspacePolicy::CachedProducts,
    CoupledPressureVelocityCacheStatistics* statistics = nullptr);

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
 * Public pressure fields are physical gauge pressure in Pa. Assembly and the
 * Krylov vector use normalized pressure `q = p / reference_density`; solve()
 * converts back to Pa when it unpacks the accepted result. The `Assembled`
 * backend builds the full coupled CRS matrix. `BlockComposite` delegates
 * application to CoupledBlockOperator while retaining assembled block
 * operators; it is not described as a generic matrix-free discretization.
 *
 * @par Algorithm
 * @if SIMPLEFLUID_DIAGRAMS
 * @startuml
 * start
 * :FluidSolver reconstructs predictor face flux;
 * :assemble(..., CoupledAssemblyPurpose::LinearSolve);
 * :Refresh geometry/adjacency caches when epoch changed;
 * :Assemble or refresh momentum system A;
 * :Validate velocity/pressure boundaries\nand volume-continuity target;
 * if (Physical pressure Dirichlet exists?) then (yes)
 *   :Use physical pressure boundary rows;
 * else (all Neumann)
 *   :Fix minimum global cell GID\nas the normalized-pressure gauge;
 * endif
 * :Assemble/refresh G, D, stabilization C, and RHS\nfor q = p / reference_density;
 * if (coupled_operator_backend == Assembled?) then (yes)
 *   :Assemble full coupled CRS operator;
 * else (BlockComposite)
 *   :Build/reuse CoupledBlockOperator\nfrom assembled block operators;
 * endif
 * :Build/refresh Schur approximation\nC - sum(D_i diag(A)^-1 G_i);
 * :Return CoupledPressureVelocitySystem;
 * :solve(system, velocity, physical pressure, options);
 * :Pack warm start [u, p/reference_density];
 * if (operator graphs reusable and\nCoupledRebuildPolicy allows?) then (yes)
 *   :Refresh numeric Schur preconditioner state;
 * else (no)
 *   :Build Jacobi momentum and\nMueLu Schur preconditioners;
 * endif
 * :Delegate inner iteration to Belos flexible block GMRES;
 * :Compute algebraic continuity diagnostics;
 * :Unpack velocity and\npressure = reference_density * q;
 * :Synchronize periodic field ghosts;
 * :Return convergence status and Krylov statistics;
 * stop
 * @enduml
 * @endif
 *
 * @see CoupledPressureVelocitySystem
 * @see CoupledRebuildPolicy
 *
 * @tparam Pack Tpetra type pack defining distributed algebra types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>>
class SIMPLEFLUID_SOLVERS_EXPORT CoupledPressureVelocitySolver
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using momentum_equation_type = IncompressibleMomentumEquation<Pack, mesh_type>;
    using boussinesq_momentum_equation_type = BoussinesqMomentumEquation<Pack, mesh_type>;
    using material_property_fields_type = MaterialPropertyFields<Pack, mesh_type>;
    using velocity_boundary_cache_type = std::conditional_t<std::same_as<mesh_type, Mesh<Pack>>,
        FVM::VelocityBoundaryCache<Pack>, FVM::FieldStoredVelocityBoundaryCache<Pack, mesh_type>>;
    using boundary_cache_type = FVM::MeshBoundaryCache<Pack, mesh_type>;
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
    using continuity_target_type = VolumeContinuityTarget<Pack, mesh_type>;
    using fixed_boundary_flux_provider_type = std::function<scalar_type(int, size_t, local_ordinal_type)>;
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

    CoupledStorageStatistics storage_statistics() const;

    /**
     * Prescribe exact owner-oriented volume fluxes [m^3/s] on named
     * physical-pressure Dirichlet boundaries.
     *
     * Selected faces are removed from the adjustable pressure-outlet part of
     * the continuity operator and their exact flux enters its affine RHS.
     */
    void set_fixed_boundary_flux_provider(
        std::vector<std::string> boundary_names, fixed_boundary_flux_provider_type provider, std::uint64_t generation);

    /** Restore ordinary pressure-outlet treatment on all boundaries. */
    void clear_fixed_boundary_flux_provider();

    /** Overwrite selected owned faces after the caller reconstructs fluxes. */
    void apply_fixed_boundary_fluxes(face_flux_field_type& fluxes) const;

    /**
     * Return a cache suitable for Rhie--Chow reconstruction before applying
     * the exact fixed fluxes. Only registered faces are presented as
     * velocity-Neumann to the pressure/velocity compatibility check.
     */
    velocity_boundary_cache_type pressure_flux_boundary_cache(const velocity_boundary_cache_type& boundary_cache) const;

    /**
     * @brief Assemble an isothermal incompressible coupled system.
     * @param momentum_equation Momentum operator and forcing data.
     * @param velocity Current velocity field.
     * @param pressure Current physical pressure in Pa.
     * @param face_fluxes Current owner-oriented face flux field.
     * @param velocity_boundary_cache Cached velocity boundary values.
     * @param boundary_conditions Boundary conditions for coupled fields.
     * @param time_options Physical timestep and discretization controls.
     * @param reference_density Positive density used to normalize pressure.
     */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, scalar_type reference_density = scalar_type{1}) const;

    /** Assemble for a selected purpose without changing the linear-solve entry point. */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, scalar_type reference_density, CoupledAssemblyPurpose purpose) const;

    /** Assemble with an integrated per-cell continuity target [m^3/s]. */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, const continuity_target_type& continuity_target,
        scalar_type reference_density = scalar_type{1}, const FVM::ALEControlVolumeState* ale = nullptr) const;

    /** Assemble a continuity-target system for a selected purpose. */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, const continuity_target_type& continuity_target,
        scalar_type reference_density, const FVM::ALEControlVolumeState* ale, CoupledAssemblyPurpose purpose) const;

    /**
     * @brief Assemble isothermal momentum with variable effective viscosity
     *        and isotropic Reynolds stress.
     * @param momentum_equation Momentum operator and forcing data.
     * @param velocity Current velocity field.
     * @param pressure Current physical pressure in Pa.
     * @param face_fluxes Current owner-oriented face flux field.
     * @param velocity_boundary_cache Cached velocity boundary values.
     * @param boundary_conditions Boundary conditions for coupled fields.
     * @param time_options Physical timestep and discretization controls.
     * @param reference_density Positive density used to normalize pressure.
     * @param dynamic_viscosity_override Optional molecular-plus-turbulent
     *        dynamic viscosity used by the momentum block.
     * @param turbulent_kinetic_energy_gradient Optional gradient contributing
     *        the isotropic Reynolds-stress acceleration.
     * @param boundary_dynamic_viscosity Optional boundary cache for dynamic viscosity.
     */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, scalar_type reference_density,
        const field_type* dynamic_viscosity_override, const velocity_field_type* turbulent_kinetic_energy_gradient,
        const boundary_cache_type* boundary_dynamic_viscosity) const;

    /** Assemble variable-viscosity momentum for a selected purpose. */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, scalar_type reference_density,
        const field_type* dynamic_viscosity_override, const velocity_field_type* turbulent_kinetic_energy_gradient,
        const boundary_cache_type* boundary_dynamic_viscosity, CoupledAssemblyPurpose purpose) const;

    /** Variable-viscosity assembly with an integrated continuity target. */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, scalar_type reference_density,
        const field_type* dynamic_viscosity_override, const velocity_field_type* turbulent_kinetic_energy_gradient,
        const boundary_cache_type* boundary_dynamic_viscosity, const continuity_target_type& continuity_target,
        const FVM::ALEControlVolumeState* ale = nullptr) const;

    /** Assemble variable-viscosity momentum and a target for a selected purpose. */
    system_type assemble(const momentum_equation_type& momentum_equation, const velocity_field_type& velocity,
        const field_type& pressure, const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache, const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& time_options, scalar_type reference_density,
        const field_type* dynamic_viscosity_override, const velocity_field_type* turbulent_kinetic_energy_gradient,
        const boundary_cache_type* boundary_dynamic_viscosity, const continuity_target_type& continuity_target,
        const FVM::ALEControlVolumeState* ale, CoupledAssemblyPurpose purpose) const;

    /**
     * @brief Assemble a thermally buoyant coupled system.
     * @param momentum_equation Boussinesq momentum operator and forcing data.
     * @param velocity Current velocity field.
     * @param pressure Current physical pressure in Pa.
     * @param temperature Current temperature field.
     * @param face_fluxes Current owner-oriented face flux field.
     * @param velocity_boundary_cache Cached velocity boundary values.
     * @param boundary_conditions Boundary conditions for coupled fields.
     * @param time_options Physical timestep and discretization controls.
     * @param material Optional frozen material-property fields.
     * @param reference_density Positive density used to normalize pressure.
     * @param density_feedback_enabled Whether density variation feeds back into momentum.
     * @param dynamic_viscosity_override Optional effective dynamic-viscosity field.
     * @param turbulent_kinetic_energy_gradient Optional isotropic Reynolds-stress gradient.
     * @param boundary_dynamic_viscosity Optional boundary cache for dynamic viscosity.
     * @throws std::invalid_argument If supplied fields are incompatible.
     */
    system_type assemble(const boussinesq_momentum_equation_type& momentum_equation,
        const velocity_field_type& velocity, const field_type& pressure, const field_type& temperature,
        const face_flux_field_type& face_fluxes, const velocity_boundary_cache_type& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        const material_property_fields_type* material = nullptr, scalar_type reference_density = scalar_type{1},
        bool density_feedback_enabled = false, const field_type* dynamic_viscosity_override = nullptr,
        const velocity_field_type* turbulent_kinetic_energy_gradient = nullptr,
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr) const;

    /** Assemble thermally buoyant momentum for a selected purpose. */
    system_type assemble(const boussinesq_momentum_equation_type& momentum_equation,
        const velocity_field_type& velocity, const field_type& pressure, const field_type& temperature,
        const face_flux_field_type& face_fluxes, const velocity_boundary_cache_type& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        const material_property_fields_type* material, scalar_type reference_density, bool density_feedback_enabled,
        const field_type* dynamic_viscosity_override, const velocity_field_type* turbulent_kinetic_energy_gradient,
        const boundary_cache_type* boundary_dynamic_viscosity, CoupledAssemblyPurpose purpose) const;

    /** Thermally buoyant assembly with an integrated continuity target. */
    system_type assemble(const boussinesq_momentum_equation_type& momentum_equation,
        const velocity_field_type& velocity, const field_type& pressure, const field_type& temperature,
        const face_flux_field_type& face_fluxes, const velocity_boundary_cache_type& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        const continuity_target_type& continuity_target, const material_property_fields_type* material = nullptr,
        scalar_type reference_density = scalar_type{1}, bool density_feedback_enabled = false,
        const field_type* dynamic_viscosity_override = nullptr,
        const velocity_field_type* turbulent_kinetic_energy_gradient = nullptr,
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr,
        const FVM::ALEControlVolumeState* ale = nullptr) const;

    /** Assemble thermally buoyant momentum and a target for a selected purpose. */
    system_type assemble(const boussinesq_momentum_equation_type& momentum_equation,
        const velocity_field_type& velocity, const field_type& pressure, const field_type& temperature,
        const face_flux_field_type& face_fluxes, const velocity_boundary_cache_type& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        const continuity_target_type& continuity_target, const material_property_fields_type* material,
        scalar_type reference_density, bool density_feedback_enabled, const field_type* dynamic_viscosity_override,
        const velocity_field_type* turbulent_kinetic_energy_gradient,
        const boundary_cache_type* boundary_dynamic_viscosity, const FVM::ALEControlVolumeState* ale,
        CoupledAssemblyPurpose purpose) const;

private:
    using pressure_graph_signature_type = std::vector<std::pair<std::string, BoundaryConditionType>>;

    struct StaticGeometryCache
    {
        BoundaryConditionMap pressure_boundaries;
        scalar_type reference_density = {};
        FVM::CellGradientScheme gradient_scheme = FVM::CellGradientScheme::LeastSquares;
        std::vector<detail::AffinePressureGradientStencil<Pack, mesh_type>> stencils;
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
    bool has_external_generation() const;
    void prepare_numeric_generation(const momentum_equation_type& equation, const TimeStepperOptions& options,
        CoupledAssemblyPurpose purpose) const;

    bool can_reuse_assembly_graph(
        const momentum_system_type& momentum, const pressure_graph_signature_type& pressure_signature) const;

    SIMPLEFLUID_SOLVERS_LOCAL
    system_type assemble_coupled_system(const momentum_system_type& momentum, const velocity_field_type& velocity,
        const field_type& pressure, const velocity_boundary_cache_type& velocity_boundary_cache,
        const BoundaryConditionSet& boundary_conditions, const TimeStepperOptions& time_options,
        scalar_type reference_density, const continuity_target_type& continuity_target,
        bool enforce_global_compatibility, CoupledAssemblyPurpose purpose) const;

    SIMPLEFLUID_SOLVERS_LOCAL
    void invalidate_cache() const;

    SIMPLEFLUID_SOLVERS_LOCAL
    void refresh_geometry_if_needed() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void refresh_cell_face_adjacency() const;

    SIMPLEFLUID_SOLVERS_LOCAL
    std::vector<std::optional<scalar_type>> evaluate_fixed_boundary_fluxes(
        const BoundaryConditionMap* pressure_boundaries) const;

public:
    /**
     * @brief Solve a coupled system and update velocity and physical pressure.
     * @param system Assembled coupled linear system.
     * @param velocity Velocity field updated on convergence or termination.
     * @param pressure Physical pressure field updated in Pa.
     * @param options Linear solver controls.
     * @return Convergence status and Krylov statistics.
     * @throws std::invalid_argument If the pressure normalization is invalid.
     * @throws std::runtime_error If the Belos problem cannot be initialized.
     */
    result_type solve(const system_type& system, velocity_field_type& velocity, field_type& pressure,
        const LinearSolverOptions& options) const;

    /** Prepare the native Schur inverse for a separate linear correction solve.
     * Retained inverse instances are not refreshed in place by default. Retain
     * the system's linear_operator too when assembling later generations, to
     * protect its shared matrix blocks. The historical linear driver can opt
     * out once its prior solve has completed.
     */
    Teuchos::RCP<const operator_type> right_preconditioner(
        const system_type& system, bool preserve_retained_generations = true) const;

private:
    SP<const mesh_type> d_mesh;
    CoupledRebuildPolicy d_rebuild_policy = CoupledRebuildPolicy::OnOperatorGraphChange;
    Teuchos::RCP<const typename Pack::map_type> d_coupled_map;
    mutable std::vector<FVM::detail::BoundaryFaceLocation<mesh_type>> d_boundary_locations;
    mutable std::uint64_t d_geometry_epoch = 0;
    mutable std::vector<size_t> d_cell_face_offsets;
    mutable std::vector<local_ordinal_type> d_cell_faces;
    mutable std::uint64_t d_cell_face_epoch = 0;
    // Cell first, then first-seen neighbours; face order is never changed.
    mutable std::vector<size_t> d_coefficient_offsets;
    mutable std::vector<local_ordinal_type> d_coefficient_columns;
    mutable std::vector<size_t> d_face_neighbor_slots;
    mutable scalar_type d_geometry_block_time_step = {};
    mutable bool d_geometry_block_linear_weights = false;
    mutable std::vector<std::optional<scalar_type>> d_geometry_block_fixed_fluxes;
    mutable CoupledPressureVelocityCacheStatistics d_cache_statistics;
    mutable system_type d_cached_system;
    mutable std::optional<std::pair<CoupledOperatorBackend, CoupledWorkspacePolicy>> d_logged_backends;
    // Weak observers include retained old generations without keeping them alive.
    mutable std::vector<Teuchos::RCP<matrix_type>> d_observed_matrices;
    mutable std::vector<Teuchos::RCP<const operator_type>> d_observed_operators;
    void observe_storage() const;
    mutable StaticGeometryCache d_static_geometry;
    mutable detail::CoupledSchurWorkspace<Pack> d_schur_workspace;
    mutable std::array<Teuchos::RCP<matrix_type>, 3> d_gradient_stabilization_products;
    mutable const graph_type* d_cached_momentum_graph = nullptr;
    mutable pressure_graph_signature_type d_cached_pressure_graph_signature;
    mutable Teuchos::RCP<preconditioner_type> d_preconditioner;
    mutable Teuchos::RCP<solver_type> d_belos_solver;
    mutable Teuchos::RCP<vector_type> d_solution;
    std::vector<std::string> d_fixed_boundary_flux_names;
    fixed_boundary_flux_provider_type d_fixed_boundary_flux_provider;
    std::uint64_t d_fixed_boundary_flux_generation = 0;
    std::uint64_t d_fixed_boundary_flux_revision = 0;
};

/**
 * @brief Library-provided coupled solver specialization.
 *
 * Additional Tpetra packs require a deliberate explicit instantiation in the
 * solver implementation target.
 */
extern template class CoupledPressureVelocitySolver<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class CoupledPressureVelocitySolver<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
