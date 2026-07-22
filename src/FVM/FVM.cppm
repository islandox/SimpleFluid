module;

#include "cmake/StandardHeaders.hh"

#include "FVM/BoundaryCache.hh"
#include "FVM/EquationOperators.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/Operators.hh"

export module SimpleFluid.FVM;

export import SimpleFluid.Fields;
export import SimpleFluid.LinearSolvers;

export namespace SimpleFluid
{
using ::SimpleFluid::BoundaryCache;
using ::SimpleFluid::BoundaryCondition;
using ::SimpleFluid::BoundaryConditionMap;
using ::SimpleFluid::BoundaryConditionSet;
using ::SimpleFluid::BoundaryConditionType;
using ::SimpleFluid::VectorBoundaryCondition;
using ::SimpleFluid::VectorBoundaryConditionMap;
using ::SimpleFluid::LinearPreconditioner;
using ::SimpleFluid::LinearSolveStatistics;
using ::SimpleFluid::LinearSolveSummary;
using ::SimpleFluid::LinearSolverOptions;
using ::SimpleFluid::BelosLinearSolver;
using ::SimpleFluid::solve_linear_system;
using ::SimpleFluid::to_string;
using ::SimpleFluid::cache_boundary_conditions;
}

export namespace SimpleFluid::FVM
{
using ::SimpleFluid::FVM::ConvectionOperator;
using ::SimpleFluid::FVM::DiffusionOperator;
using ::SimpleFluid::FVM::DiffusionSystem;
using ::SimpleFluid::FVM::DivergenceOperator;
using ::SimpleFluid::FVM::GradientOperator;
using ::SimpleFluid::FVM::NonOrthogonalTreatment;
using ::SimpleFluid::FVM::OperatorVariant;
using ::SimpleFluid::FVM::SourceOperator;
using ::SimpleFluid::FVM::TransientOperator;
using ::SimpleFluid::FVM::TransportSystem;
using ::SimpleFluid::FVM::VectorDiffusionSystem;
using ::SimpleFluid::FVM::VectorTransportSystem;
using ::SimpleFluid::FVM::VelocityBoundaryCache;
using ::SimpleFluid::FVM::add_explicit_non_orthogonal_correction;
using ::SimpleFluid::FVM::cache_velocity_boundary_conditions;
using ::SimpleFluid::FVM::cell_divergence_from_fluxes;
using ::SimpleFluid::FVM::cell_flux_balance;
using ::SimpleFluid::FVM::cell_gradient;
using ::SimpleFluid::FVM::diffusion_matrix;
using ::SimpleFluid::FVM::diffusion_system;
using ::SimpleFluid::FVM::explicit_non_orthogonal_diffusion_system;
using ::SimpleFluid::FVM::face_fluxes;
using ::SimpleFluid::FVM::face_velocities;
using ::SimpleFluid::FVM::fully_implicit_non_orthogonal_diffusion_system;
using ::SimpleFluid::FVM::full_diffusion_residual;
using ::SimpleFluid::FVM::implicit_non_orthogonal_diffusion_system;
using ::SimpleFluid::FVM::identity_matrix;
using ::SimpleFluid::FVM::non_orthogonal_diffusion_system;
using ::SimpleFluid::FVM::non_orthogonal_transport_system;
using ::SimpleFluid::FVM::normal_face_fluxes;
using ::SimpleFluid::FVM::non_orthogonal_treatment_from_string;
using ::SimpleFluid::FVM::physical_momentum_transport_system;
using ::SimpleFluid::FVM::physical_temperature_transport_system;
using ::SimpleFluid::FVM::pressure_poisson_matrix;
using ::SimpleFluid::FVM::pressure_weighted_face_fluxes;
using ::SimpleFluid::FVM::solve_explicit_non_orthogonal_diffusion;
using ::SimpleFluid::FVM::solve_non_orthogonal_diffusion;
using ::SimpleFluid::FVM::slip_face_velocity;
using ::SimpleFluid::FVM::to_string;
using ::SimpleFluid::FVM::transport_system;
using ::SimpleFluid::FVM::upwind_convection_matrix;
using ::SimpleFluid::FVM::vector_diffusion_system;
using ::SimpleFluid::FVM::weighted_scalar_transport_system;
}
