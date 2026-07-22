module;

#include "cmake/StandardHeaders.hh"

#include "solvers/BelosLinearSolver.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/PressureVelocityCoupling.hh"
#include "equations/TimeStepperOptions.hh"
#include "equations/EquationValidation.hh"
#include "equations/Equation.hh"
#include "equations/AssembledEquation.hh"
#include "equations/CoupledEquation.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/FissionPowerSource.hh"
#include "equations/BoilingSourceModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/DelayedNeutronPrecursorModel.hh"
#include "equations/FeedbackMap.hh"
#include "equations/RadiolyticGasProperties.hh"
#include "equations/RadiolyticGasModel.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"

export module SimpleFluid.Equations;

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
using ::SimpleFluid::boussinesq_model_options_from_database;
using ::SimpleFluid::BoussinesqUpdateContext;
using ::SimpleFluid::initialize_cell_field;
using ::SimpleFluid::VolumetricScalarSource;
using ::SimpleFluid::TemperatureSourceRegistry;
using ::SimpleFluid::MaterialPropertyFields;
using ::SimpleFluid::SolutionOutputOptions;

using ::SimpleFluid::FissionPowerProfile;
using ::SimpleFluid::fission_power_profile_from_string;
using ::SimpleFluid::FissionPowerSourceOptions;
using ::SimpleFluid::fission_power_source_options_from_database;
using ::SimpleFluid::FissionPowerSource;

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

using ::SimpleFluid::DelayedNeutronPrecursorOptions;
using ::SimpleFluid::validate_delayed_neutron_precursor_options;
using ::SimpleFluid::delayed_neutron_precursor_options_from_database;
using ::SimpleFluid::DelayedNeutronPrecursorModel;

using ::SimpleFluid::RadiolyticGasMode;
using ::SimpleFluid::RadiolyticPressureMode;
using ::SimpleFluid::RadiolyticTransportMode;
using ::SimpleFluid::BubbleTransportMode;
using ::SimpleFluid::RadiolyticHeavisideMode;
using ::SimpleFluid::BubbleRiseVelocityMode;
using ::SimpleFluid::SurfaceTensionMode;
using ::SimpleFluid::HydrogenDiffusivityMode;
using ::SimpleFluid::RadiolyticGasOptions;
using ::SimpleFluid::radiolytic_gas_options_from_database;
using ::SimpleFluid::validate_radiolytic_gas_options;
using ::SimpleFluid::RadiolyticGasStepStatistics;
using ::SimpleFluid::RadiolyticGasModel;

using ::SimpleFluid::IncompressibleMomentumEquation;
using ::SimpleFluid::BoussinesqMomentumEquation;
using ::SimpleFluid::pressure_projection_linear_solver_options;
using ::SimpleFluid::PressureProjectionEquation;
using ::SimpleFluid::TemperatureDiffusionEquation;
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

export namespace SimpleFluid::RadiolyticGasPhysics
{
using ::SimpleFluid::RadiolyticGasPhysics::require_positive;
using ::SimpleFluid::RadiolyticGasPhysics::require_non_negative;
using ::SimpleFluid::RadiolyticGasPhysics::ideal_gas_alpha_source;
using ::SimpleFluid::RadiolyticGasPhysics::henry_equilibrium_concentration;
using ::SimpleFluid::RadiolyticGasPhysics::pressure_nucleation_correction;
using ::SimpleFluid::RadiolyticGasPhysics::mean_fission_fragment_let;
using ::SimpleFluid::RadiolyticGasPhysics::pure_water_nucleation_radius;
using ::SimpleFluid::RadiolyticGasPhysics::atmospheric_nucleation_radius;
using ::SimpleFluid::RadiolyticGasPhysics::sheng2024_nucleation_radius;
using ::SimpleFluid::RadiolyticGasPhysics::sheng2024_surface_tension;
using ::SimpleFluid::RadiolyticGasPhysics::sheng2024_hydrogen_diffusivity;
using ::SimpleFluid::RadiolyticGasPhysics::hughmark_sherwood;
using ::SimpleFluid::RadiolyticGasPhysics::hughmark_mass_transfer_coefficient;
using ::SimpleFluid::RadiolyticGasPhysics::celata2007_drag_coefficient;
using ::SimpleFluid::RadiolyticGasPhysics::BubbleRiseVelocityResult;
using ::SimpleFluid::RadiolyticGasPhysics::celata2007_bubble_rise_velocity;
using ::SimpleFluid::RadiolyticGasPhysics::BubbleRadiusResult;
using ::SimpleFluid::RadiolyticGasPhysics::solve_bubble_radius;
using ::SimpleFluid::RadiolyticGasPhysics::bubble_void_fraction;
using ::SimpleFluid::RadiolyticGasPhysics::characteristic_radius;
using ::SimpleFluid::RadiolyticGasPhysics::smoothed_heaviside;
}
