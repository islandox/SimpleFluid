module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "solvers/BelosLinearSolver.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/PressureVelocityCoupling.hh"
#include "equations/TimeStepperOptions.hh"
#include "equations/EquationValidation.hh"
#include "equations/Equation.hh"
#include "equations/AssembledEquation.hh"
#include "equations/CoupledEquation.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/PrescribedHeatSource.hh"
#include "equations/BoilingSourceModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/ScalarSourceDecayTransport.hh"
#include "equations/FeedbackMap.hh"
#include "equations/ReactingGasProperties.hh"
#include "equations/ReactingGasModel.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "equations/turbulence/TurbulenceModel.hh"
#include "equations/turbulence/WallDistanceEquation.hh"

export module SimpleFluid.Equations;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.FVM;

export namespace SimpleFluid
{
using ::SimpleFluid::LinearPreconditioner;
using ::SimpleFluid::to_string;
using ::SimpleFluid::LinearSolveStatistics;
using ::SimpleFluid::LinearSolveSummary;
using ::SimpleFluid::LinearSolverOptions;
using ::SimpleFluid::BelosLinearSolver;
using ::SimpleFluid::solve_linear_system;

using ::SimpleFluid::PressureVelocityCoupling;
using ::SimpleFluid::pressure_velocity_coupling_from_string;
using ::SimpleFluid::PressureVelocityResiduals;
using ::SimpleFluid::FluidStepStatistics;
using ::SimpleFluid::BoussinesqStepStatistics;
using ::SimpleFluid::TimeStepperOptions;

using ::SimpleFluid::Equation;
using ::SimpleFluid::AssembledEquation;
using ::SimpleFluid::AssembledEquationConcept;
using ::SimpleFluid::AssembledEquationModel;
using ::SimpleFluid::AssembledCoupledEquation;
using ::SimpleFluid::CoupledSolverBackend;
using ::SimpleFluid::IndependentBlockSolver;
using ::SimpleFluid::CoupledEquation;
using ::SimpleFluid::make_assembled_equation_model;

using ::SimpleFluid::BoussinesqModelOptions;
using ::SimpleFluid::validate_boussinesq_model_options;
using ::SimpleFluid::boussinesq_model_options_from_database;
using ::SimpleFluid::BoussinesqUpdateContext;
using ::SimpleFluid::initialize_cell_field;
using ::SimpleFluid::VolumetricScalarSource;
using ::SimpleFluid::TemperatureSourceRegistry;
using ::SimpleFluid::MaterialPropertyFields;
using ::SimpleFluid::SolutionOutputOptions;

using ::SimpleFluid::PrescribedHeatProfile;
using ::SimpleFluid::prescribed_heat_profile_from_string;
using ::SimpleFluid::PrescribedHeatSourceOptions;
using ::SimpleFluid::validate_prescribed_heat_power_options;
using ::SimpleFluid::prescribed_heat_source_options_from_database;
using ::SimpleFluid::PrescribedHeatSource;

using ::SimpleFluid::BoilingSourceOptions;
using ::SimpleFluid::validate_boiling_source_options;
using ::SimpleFluid::boiling_source_options_from_database;
using ::SimpleFluid::BoilingSourceModel;

using ::SimpleFluid::ScalarVoidFractionOptions;
using ::SimpleFluid::validate_scalar_void_fraction_options;
using ::SimpleFluid::scalar_void_fraction_options_from_database;
using ::SimpleFluid::ScalarVoidFractionModel;

using ::SimpleFluid::DensityFeedbackMode;
using ::SimpleFluid::ViscosityFeedbackMode;
using ::SimpleFluid::MaterialFeedbackOptions;
using ::SimpleFluid::parse_density_feedback_mode;
using ::SimpleFluid::parse_viscosity_feedback_mode;
using ::SimpleFluid::validate_material_feedback_options;
using ::SimpleFluid::material_feedback_options_from_database;
using ::SimpleFluid::MaterialFeedbackModel;

using ::SimpleFluid::ScalarSourceDecayOptions;
using ::SimpleFluid::validate_scalar_source_decay_options;
using ::SimpleFluid::scalar_source_decay_options_from_database;
using ::SimpleFluid::ScalarSourceDecayTransport;

using ::SimpleFluid::ReactingGasMode;
using ::SimpleFluid::ReactingPressureMode;
using ::SimpleFluid::ReactingTransportMode;
using ::SimpleFluid::BubbleTransportMode;
using ::SimpleFluid::ReactingHeavisideMode;
using ::SimpleFluid::BubbleRiseVelocityMode;
using ::SimpleFluid::SurfaceTensionMode;
using ::SimpleFluid::HydrogenDiffusivityMode;
using ::SimpleFluid::ReactingGasOptions;
using ::SimpleFluid::reacting_gas_options_from_database;
using ::SimpleFluid::validate_reacting_gas_options;
using ::SimpleFluid::ReactingGasStepStatistics;
using ::SimpleFluid::ReactingGasModel;

using ::SimpleFluid::IncompressibleMomentumEquation;
using ::SimpleFluid::BoussinesqMomentumEquation;
using ::SimpleFluid::pressure_projection_linear_solver_options;
using ::SimpleFluid::PressureProjectionEquation;
using ::SimpleFluid::TemperatureDiffusionEquation;

using ::SimpleFluid::BSLKOmegaEquation;
using ::SimpleFluid::RNGKEpsilonEquation;
using ::SimpleFluid::RealizableKEpsilonEquation;
using ::SimpleFluid::SSTKOmegaEquation;
using ::SimpleFluid::StandardKEpsilonEquation;
using ::SimpleFluid::StandardKOmegaEquation;
using ::SimpleFluid::TurbulenceModelType;
using ::SimpleFluid::parse_turbulence_model_type;
using ::SimpleFluid::TurbulenceWallTreatmentType;
using ::SimpleFluid::parse_turbulence_wall_treatment_type;
using ::SimpleFluid::TurbulenceBuoyancyModel;
using ::SimpleFluid::parse_turbulence_buoyancy_model;
using ::SimpleFluid::TurbulenceModelOptions;
using ::SimpleFluid::validate_turbulence_model_options;
using ::SimpleFluid::turbulence_model_options_from_database;
using ::SimpleFluid::TurbulenceScalarBoundaryOverrides;
using ::SimpleFluid::TurbulenceScalarTransportEquation;
using ::SimpleFluid::TurbulenceBuoyancyContext;
using ::SimpleFluid::TurbulenceWallRoughnessModel;
using ::SimpleFluid::parse_turbulence_wall_roughness_model;
using ::SimpleFluid::TurbulenceThermalWallLaw;
using ::SimpleFluid::parse_turbulence_thermal_wall_law;
using ::SimpleFluid::TurbulenceWallTreatmentOptions;
using ::SimpleFluid::validate_turbulence_wall_treatment_options;
using ::SimpleFluid::openfoam_y_plus_lam;
using ::SimpleFluid::ResolvedLowReSSTWallPolicy;
using ::SimpleFluid::ResolvedLowReKEpsilonWallPolicy;
using ::SimpleFluid::StandardHighReKEpsilonWallPolicy;
using ::SimpleFluid::TurbulenceWallFaceEvaluation;
using ::SimpleFluid::TurbulenceWallTreatment;
using ::SimpleFluid::ResolvedLowReSSTWallTreatment;
using ::SimpleFluid::ResolvedLowReKEpsilonWallTreatment;
using ::SimpleFluid::StandardHighReKEpsilonWallTreatment;
using ::SimpleFluid::TurbulenceModel;
using ::SimpleFluid::WallDistanceEquationOptions;
using ::SimpleFluid::validate_wall_distance_equation_options;
using ::SimpleFluid::PoissonWallDistanceEquation;
}

export namespace SimpleFluid::EquationValidation
{
using ::SimpleFluid::EquationValidation::require_non_null_mesh;
using ::SimpleFluid::EquationValidation::require_mesh_match;
using ::SimpleFluid::EquationValidation::require_non_negative;
using ::SimpleFluid::EquationValidation::assert_sufficient_cache_size;
}

export namespace SimpleFluid::FeedbackMap
{
using ::SimpleFluid::FeedbackMap::FeedbackCell;
using ::SimpleFluid::FeedbackMap::volume_weighted_average;
using ::SimpleFluid::FeedbackMap::import_power_density;
}

export namespace SimpleFluid::BubbleGasPhysics
{
using ::SimpleFluid::BubbleGasPhysics::require_positive;
using ::SimpleFluid::BubbleGasPhysics::require_non_negative;
using ::SimpleFluid::BubbleGasPhysics::ideal_gas_alpha_source;
using ::SimpleFluid::BubbleGasPhysics::henry_equilibrium_concentration;
using ::SimpleFluid::BubbleGasPhysics::pressure_nucleation_correction;
using ::SimpleFluid::BubbleGasPhysics::winter2020_reference_let;
using ::SimpleFluid::BubbleGasPhysics::pure_water_nucleation_radius;
using ::SimpleFluid::BubbleGasPhysics::atmospheric_nucleation_radius;
using ::SimpleFluid::BubbleGasPhysics::sheng2024_nucleation_radius;
using ::SimpleFluid::BubbleGasPhysics::sheng2024_surface_tension;
using ::SimpleFluid::BubbleGasPhysics::sheng2024_hydrogen_diffusivity;
using ::SimpleFluid::BubbleGasPhysics::hughmark_sherwood;
using ::SimpleFluid::BubbleGasPhysics::hughmark_mass_transfer_coefficient;
using ::SimpleFluid::BubbleGasPhysics::celata2007_drag_coefficient;
using ::SimpleFluid::BubbleGasPhysics::BubbleRiseVelocityResult;
using ::SimpleFluid::BubbleGasPhysics::celata2007_bubble_rise_velocity;
using ::SimpleFluid::BubbleGasPhysics::BubbleRadiusResult;
using ::SimpleFluid::BubbleGasPhysics::solve_bubble_radius;
using ::SimpleFluid::BubbleGasPhysics::bubble_void_fraction;
using ::SimpleFluid::BubbleGasPhysics::characteristic_radius;
using ::SimpleFluid::BubbleGasPhysics::smoothed_heaviside;
}

// Implementation units need these Teuchos names while defining the exported
// equation templates.  Re-export only that narrow surface instead of making
// the whole Trilinos header-unit facade a dependency of this project module.
export namespace Teuchos
{
using ::Teuchos::Array;
using ::Teuchos::RCP;
using ::Teuchos::null;
using ::Teuchos::rcp;
}
