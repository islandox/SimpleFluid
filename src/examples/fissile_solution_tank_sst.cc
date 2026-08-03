/**
 * @file fissile_solution_tank_sst.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief OpenFOAM-comparison case for a Gaussian-heated fissile-solution tank.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FVM/OperatorDetails.hh"
#include "dataclass/Database.hh"
#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/SolverProgress.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

int positive_environment_integer(const char* name, int fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
        return fallback;

    const int value = std::stoi(text);
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive.");
    return value;
}

double positive_environment_real(const char* name, double fallback)
{
    const char* text = std::getenv(name);
    if (text == nullptr)
        return fallback;

    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value))
    {
        throw std::invalid_argument(std::string(name) + " must be finite and positive.");
    }
    return value;
}

std::vector<double> graded_tank_edges(double length, double nominal_spacing, int boundary_layer_count,
    double first_height, double growth, bool lower_boundary_layer, bool upper_boundary_layer)
{
    const auto base_cell_count = static_cast<size_t>(std::ceil(length / nominal_spacing));
    const auto lower_count = lower_boundary_layer ? static_cast<size_t>(boundary_layer_count) : 0;
    const auto upper_count = upper_boundary_layer ? static_cast<size_t>(boundary_layer_count) : 0;
    if (lower_count + upper_count >= base_cell_count)
    {
        throw std::invalid_argument("Tank boundary-layer counts overlap.");
    }

    std::vector<double> layer_widths(static_cast<size_t>(boundary_layer_count));
    double width = first_height;
    for (auto& layer_width : layer_widths)
    {
        layer_width = width;
        width *= growth;
    }
    const double lower_thickness =
        lower_boundary_layer ? std::accumulate(layer_widths.begin(), layer_widths.end(), 0.0) : 0.0;
    const double upper_thickness =
        upper_boundary_layer ? std::accumulate(layer_widths.begin(), layer_widths.end(), 0.0) : 0.0;
    const auto interior_count = base_cell_count - lower_count - upper_count;
    const double interior_width = (length - lower_thickness - upper_thickness) / static_cast<double>(interior_count);
    if (!(interior_width > 0.0))
    {
        throw std::invalid_argument("Tank boundary-layer thicknesses overlap.");
    }

    std::vector<double> edges;
    edges.reserve(base_cell_count + 1);
    edges.push_back(0.0);
    if (lower_boundary_layer)
    {
        for (const auto layer_width : layer_widths)
            edges.push_back(edges.back() + layer_width);
    }
    for (size_t cell = 0; cell < interior_count; ++cell)
        edges.push_back(edges.back() + interior_width);
    if (upper_boundary_layer)
    {
        for (auto iter = layer_widths.rbegin(); iter != layer_widths.rend(); ++iter)
        {
            edges.push_back(edges.back() + *iter);
        }
    }
    edges.back() = length;
    return edges;
}

size_t tank_bin_index(double coordinate, const std::vector<double>& edges)
{
    const double tolerance = 1.0e-10 * std::max(1.0, std::abs(edges.back()));
    if (coordinate < edges.front() - tolerance || coordinate > edges.back() + tolerance)
    {
        throw std::runtime_error("Tank cell center lies outside the R-Z projection grid.");
    }
    const auto upper = std::upper_bound(edges.begin(), edges.end(), coordinate);
    const auto index =
        upper == edges.begin() ? size_t{0} : static_cast<size_t>(std::distance(edges.begin(), upper) - 1);
    return std::min(index, edges.size() - 2);
}

/**
 * @brief Project the full-cylinder state onto a volume-weighted R-Z mean.
 */
template<class Pack>
void project_axisymmetric_tank_state(SimpleFluid::BoussinesqSolver<Pack>& solver,
    SimpleFluid::TurbulenceModel<Pack>& turbulence, const std::vector<double>& radial_edges,
    const std::vector<double>& axial_edges, double reference_density)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vector_type = typename SimpleFluid::VectorCellField<Pack>::vec_type;

    constexpr size_t volume_component = 0;
    constexpr size_t temperature_component = 1;
    constexpr size_t radial_velocity_component = 2;
    constexpr size_t axial_velocity_component = 3;
    constexpr size_t pressure_component = 4;
    constexpr size_t k_component = 5;
    constexpr size_t omega_component = 6;
    constexpr size_t nut_component = 7;
    constexpr size_t component_count = 8;

    const auto mesh = solver.temperature().mesh_ptr();
    const auto radial_cells = radial_edges.size() - 1;
    const auto axial_cells = axial_edges.size() - 1;
    const auto group_count = radial_cells * axial_cells;
    if (group_count > static_cast<size_t>(std::numeric_limits<int>::max()) / component_count)
    {
        throw std::overflow_error("Tank axisymmetric projection reduction is too large.");
    }
    std::vector<scalar_type> local(group_count * component_count, scalar_type{});
    std::vector<scalar_type> global(group_count * component_count, scalar_type{});
    const auto* omega = turbulence.specific_dissipation_rate();
    if (omega == nullptr)
    {
        throw std::logic_error("Tank axisymmetric projection requires SST omega.");
    }

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        const auto radial = tank_bin_index(std::hypot(center.x, center.y), radial_edges);
        const auto axial = tank_bin_index(center.z, axial_edges);
        const auto offset = component_count * (radial + radial_cells * axial);
        const auto cell_volume = mesh->cell_volume(cell_lid);
        const auto theta = std::atan2(center.y, center.x);
        const auto cosine = std::cos(theta);
        const auto sine = std::sin(theta);
        const auto velocity = solver.velocity().value(cell_lid);

        local[offset + volume_component] += cell_volume;
        local[offset + temperature_component] += cell_volume * solver.temperature().value(cell_lid);
        local[offset + radial_velocity_component] += cell_volume * (velocity.x * cosine + velocity.y * sine);
        local[offset + axial_velocity_component] += cell_volume * velocity.z;
        local[offset + pressure_component] += cell_volume * solver.pressure().value(cell_lid);
        local[offset + k_component] += cell_volume * turbulence.turbulent_kinetic_energy().value(cell_lid);
        local[offset + omega_component] += cell_volume * omega->value(cell_lid);
        local[offset + nut_component] += cell_volume * turbulence.turbulent_kinematic_viscosity().value(cell_lid);
    }

    const auto communicator = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, static_cast<int>(local.size()), local.data(), global.data());
    for (size_t group = 0; group < group_count; ++group)
    {
        const auto volume = global[component_count * group + volume_component];
        if (!(volume > scalar_type{}) || !std::isfinite(volume))
        {
            throw std::runtime_error("Tank axisymmetric projection found an empty R-Z bin.");
        }
    }

    SimpleFluid::CellField<Pack> projected_k(mesh, "tank_axisymmetric_k");
    SimpleFluid::CellField<Pack> projected_omega(mesh, "tank_axisymmetric_omega");
    SimpleFluid::CellField<Pack> projected_nut(mesh, "tank_axisymmetric_nut");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        const auto radial = tank_bin_index(std::hypot(center.x, center.y), radial_edges);
        const auto axial = tank_bin_index(center.z, axial_edges);
        const auto offset = component_count * (radial + radial_cells * axial);
        const auto inverse_volume = scalar_type{1} / global[offset + volume_component];
        const auto radial_velocity = global[offset + radial_velocity_component] * inverse_volume;
        const auto theta = std::atan2(center.y, center.x);
        const auto raw_projected_k = global[offset + k_component] * inverse_volume;
        const auto raw_projected_omega = global[offset + omega_component] * inverse_volume;
        const auto raw_projected_nut = global[offset + nut_component] * inverse_volume;
        if (!std::isfinite(raw_projected_k) || !std::isfinite(raw_projected_omega) || !std::isfinite(raw_projected_nut))
        {
            throw std::runtime_error("Tank axisymmetric projection produced non-finite "
                                     "turbulence data.");
        }
        const auto projected_k_value =
            std::max(static_cast<scalar_type>(turbulence.options().min_turbulent_kinetic_energy), raw_projected_k);
        const auto projected_omega_value =
            std::max(static_cast<scalar_type>(turbulence.options().min_specific_dissipation_rate), raw_projected_omega);
        const auto projected_nut_value = std::max(scalar_type{}, raw_projected_nut);

        solver.temperature().set_owned_value(cell_lid, global[offset + temperature_component] * inverse_volume);
        solver.velocity().set_owned_value(
            cell_lid, vector_type{radial_velocity * std::cos(theta), radial_velocity * std::sin(theta),
                          global[offset + axial_velocity_component] * inverse_volume});
        solver.pressure().set_owned_value(cell_lid, global[offset + pressure_component] * inverse_volume);
        projected_k.set_owned_value(cell_lid, projected_k_value);
        projected_omega.set_owned_value(cell_lid, projected_omega_value);
        projected_nut.set_owned_value(cell_lid, projected_nut_value);
    }
    solver.temperature().sync_ghosts();
    solver.velocity().sync_ghosts();
    solver.pressure().sync_ghosts();
    projected_k.sync_ghosts();
    projected_omega.sync_ghosts();
    projected_nut.sync_ghosts();
    turbulence.restore_transported_state(projected_k, projected_omega, projected_nut, solver.velocity(),
        solver.material_properties(), static_cast<scalar_type>(reference_density));
}

/**
 * @brief Write rank-local cell data used by the R-Z comparison renderer.
 */
template<class Pack>
void write_profile_cells(const SimpleFluid::MeshHandle<Pack>& mesh, const SimpleFluid::BoussinesqSolver<Pack>& solver,
    const SimpleFluid::TurbulenceModel<Pack>& turbulence, const SimpleFluid::FissionPowerSource<Pack>& fission,
    int rank)
{
    const char* configured_prefix = std::getenv("SIMPLEFLUID_TANK_OUTPUT_PREFIX");
    const std::string prefix = configured_prefix == nullptr ? "simplefluid_cells" : configured_prefix;
    const std::string filename = prefix + "_rank" + std::to_string(rank) + ".csv";
    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("Cannot open " + filename + " for writing.");

    const auto* omega = turbulence.specific_dissipation_rate();
    const auto* wall_distance = turbulence.wall_distance();
    const auto* wall_y_plus = turbulence.wall_y_plus();
    if (omega == nullptr || wall_distance == nullptr || wall_y_plus == nullptr)
    {
        throw std::logic_error("The tank comparison requires SST omega and wall diagnostics.");
    }

    output << "cell_geometry_gid,time,r,theta,z,cell_volume,temperature,"
              "ur,utheta,uz,pressure,k,omega,nut,alphat,qdot_fission,"
              "wall_distance,wall_y_plus\n";
    output << std::setprecision(17);
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto center = mesh.cell_centroid(lid);
        const auto velocity = solver.velocity().value(lid);
        const double theta = std::atan2(center.y, center.x);
        const double cosine = std::cos(theta);
        const double sine = std::sin(theta);
        const double turbulent_viscosity = turbulence.turbulent_kinematic_viscosity().value(lid);
        output << mesh.cell_geometry_global_id(lid) << ',' << solver.time() << ',' << std::hypot(center.x, center.y)
               << ',' << theta << ',' << center.z << ',' << mesh.cell_volume(lid) << ','
               << solver.temperature().value(lid) << ',' << velocity.x * cosine + velocity.y * sine << ','
               << -velocity.x * sine + velocity.y * cosine << ',' << velocity.z << ',' << solver.pressure().value(lid)
               << ',' << turbulence.turbulent_kinetic_energy().value(lid) << ',' << omega->value(lid) << ','
               << turbulent_viscosity << ',' << turbulent_viscosity / turbulence.options().turbulent_prandtl_number
               << ',' << fission.field().value(lid) << ',' << wall_distance->value(lid) << ','
               << wall_y_plus->value(lid) << '\n';
    }
}

} // namespace

/**
 * @brief Run the matched Gaussian tank SST verification case.
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    using Pack = SimpleFluid::DefaultTpetraTypes;

    constexpr double radius = 0.1;
    constexpr double height = 0.3;
    constexpr double reference_temperature = 300.0;
    constexpr double density = 1000.0;
    constexpr double specific_heat_capacity = 4200.0;
    constexpr double dynamic_viscosity = 1.0e-3;
    constexpr double thermal_conductivity = 0.6;
    constexpr double thermal_expansion = 2.1e-4;
    constexpr double total_power = 1000.0;
    constexpr double gaussian_radial_width = 0.03;
    constexpr double gaussian_axial_width = 0.075;
    constexpr double initial_k = 1.0e-6;
    constexpr double initial_omega = 1.0;
    constexpr double turbulent_prandtl = 0.85;

    const double rz_spacing = positive_environment_real("SIMPLEFLUID_TANK_RZ_SPACING", 0.002);
    const double circumferential_spacing = positive_environment_real("SIMPLEFLUID_TANK_CIRCUMFERENTIAL_SPACING", 0.01);
    const int boundary_layer_count = positive_environment_integer("SIMPLEFLUID_TANK_BOUNDARY_LAYER_COUNT", 5);
    const double boundary_layer_first_height =
        positive_environment_real("SIMPLEFLUID_TANK_BOUNDARY_LAYER_FIRST_HEIGHT", 1.0e-3);
    const double boundary_layer_growth = positive_environment_real("SIMPLEFLUID_TANK_BOUNDARY_LAYER_GROWTH", 1.18);
    const int steps = positive_environment_integer("SIMPLEFLUID_TANK_STEPS", 1200);
    const double time_step = positive_environment_real("SIMPLEFLUID_TANK_DT", 0.1);
    const auto radial_projection_edges = graded_tank_edges(
        radius, rz_spacing, boundary_layer_count, boundary_layer_first_height, boundary_layer_growth, false, true);
    const auto axial_projection_edges = graded_tank_edges(
        height, rz_spacing, boundary_layer_count, boundary_layer_first_height, boundary_layer_growth, true, true);

    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", rz_spacing);
    database->set("domain_type", static_cast<int>(SimpleFluid::MeshFactory::DomainType::CYLINDER));
    database->set("radius", radius);
    database->set("height", height);
    database->set("cylinder_circumferential_mesh_size", circumferential_spacing);
    database->set("domain_exterior_face_types", SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    database->set("boundary_layer_boundary_names", SimpleFluid::ArrString{"radial", "zmin", "zmax"});
    database->set(
        "boundary_layer_counts", SimpleFluid::ArrInt{boundary_layer_count, boundary_layer_count, boundary_layer_count});
    database->set("boundary_layer_first_cell_heights",
        SimpleFluid::ArrReal{boundary_layer_first_height, boundary_layer_first_height, boundary_layer_first_height});
    database->set("boundary_layer_growth_ratios",
        SimpleFluid::ArrReal{boundary_layer_growth, boundary_layer_growth, boundary_layer_growth});
    auto mesh = SimpleFluid::MeshFactory(database).build_handle<Pack>();

    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"radial", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Dirichlet, reference_temperature};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = time_step;
    time_options.steps = steps;
    time_options.kinematic_viscosity = dynamic_viscosity / density;
    time_options.thermal_diffusivity = thermal_conductivity / (density * specific_heat_capacity);
    time_options.reference_temperature = reference_temperature;
    time_options.thermal_expansion = thermal_expansion;
    time_options.gravity_z = -9.81;
    time_options.coefficient_interpolation = SimpleFluid::FVM::FaceCoefficientInterpolation::Linear;
    time_options.pressure_gradient_scheme = SimpleFluid::FVM::CellGradientScheme::GaussLinear;
    time_options.n_pressure_correctors = 2;
    time_options.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::PISO;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1000;
    linear_options.tolerance = 1.0e-9;
    linear_options.backend = SimpleFluid::LinearSolverBackend::Gmres;
    linear_options.preconditioner = SimpleFluid::LinearPreconditioner::ILUT;
    linear_options.reuse_preconditioner = false;

    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = density;
    model_options.density = density;
    model_options.specific_heat_capacity = specific_heat_capacity;
    model_options.dynamic_viscosity = dynamic_viscosity;
    model_options.thermal_conductivity = thermal_conductivity;

    SimpleFluid::BoussinesqSolver<Pack> solver(mesh, boundaries, time_options, linear_options, model_options);
    auto pressure_options = solver.pressure_linear_solver_options();
    pressure_options.backend = SimpleFluid::LinearSolverBackend::BiCGStab;
    pressure_options.preconditioner = SimpleFluid::LinearPreconditioner::MueLu;
    pressure_options.max_iterations = 2000;
    pressure_options.tolerance = 1.0e-8;
    pressure_options.reuse_preconditioner = true;
    solver.set_pressure_linear_solver_options(pressure_options);
    solver.initialize_heated_box(reference_temperature, reference_temperature);

    SimpleFluid::FissionPowerSourceOptions fission_options;
    fission_options.profile = SimpleFluid::FissionPowerProfile::Gaussian;
    fission_options.total_power = total_power;
    fission_options.center = {0.0, 0.0, 0.5 * height};
    fission_options.standard_deviation = {gaussian_radial_width, gaussian_radial_width, gaussian_axial_width};
    solver.configure_fission_power_source(fission_options);
    auto* fission_pointer = solver.find_fission_power_source();
    if (fission_pointer == nullptr)
        throw std::logic_error("Failed to configure the Gaussian fission source.");
    auto& fission = *fission_pointer;

    SimpleFluid::TurbulenceModelOptions turbulence_options;
    turbulence_options.model = SimpleFluid::TurbulenceModelType::SSTKOmega;
    turbulence_options.initial_turbulent_kinetic_energy = initial_k;
    turbulence_options.initial_specific_dissipation_rate = initial_omega;
    turbulence_options.min_specific_dissipation_rate = 0.5;
    turbulence_options.turbulent_prandtl_number = turbulent_prandtl;
    turbulence_options.gradient_scheme = SimpleFluid::FVM::CellGradientScheme::GaussLinear;
    turbulence_options.coefficient_interpolation = SimpleFluid::FVM::FaceCoefficientInterpolation::Linear;
    turbulence_options.buoyancy_model = SimpleFluid::TurbulenceBuoyancyModel::OpenFOAMBoussinesq;
    // Supply a positive placeholder during model construction, then replace
    // it atomically with the exact cylinder distance below.
    turbulence_options.initial_wall_distance = radius;
    turbulence_options.wall_treatment = SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReSST;
    turbulence_options.wall_options.boundary_names = {"radial", "zmin", "zmax"};
    auto& turbulence = solver.configure_turbulence(turbulence_options);
    const auto field_mesh = turbulence.turbulent_kinetic_energy().mesh_ptr();
    double local_minimum_wall_face_distance = std::numeric_limits<double>::infinity();
    double local_maximum_wall_face_distance = 0.0;
    for (const auto& [batch_id, batch] : field_mesh->boundary_batches())
    {
        const auto& boundary_name = field_mesh->boundary_batch_name(batch_id);
        if (boundary_name != "radial" && boundary_name != "zmin" && boundary_name != "zmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            const auto cell_lid = field_mesh->owner_cell(face_lid);
            const double distance = SimpleFluid::FVM::detail::boundary_normal_distance(*field_mesh, face_lid, cell_lid);
            local_minimum_wall_face_distance = std::min(local_minimum_wall_face_distance, distance);
            local_maximum_wall_face_distance = std::max(local_maximum_wall_face_distance, distance);
        }
    }
    SimpleFluid::CellField<Pack> exact_wall_distance(field_mesh, "tank_exact_wall_distance");
    for (size_t owned = 0; owned < field_mesh->num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto center = field_mesh->cell_centroid(lid);
        const auto distance = std::min({radius - std::hypot(center.x, center.y), center.z, height - center.z});
        if (!(distance > 0.0) || !std::isfinite(distance))
        {
            throw std::runtime_error("Tank analytic wall distance is not finite and positive.");
        }
        exact_wall_distance.set_owned_value(lid, distance);
    }
    exact_wall_distance.sync_ghosts();
    turbulence.set_wall_distance(exact_wall_distance, solver.material_properties(), density);
    SimpleFluid::CellField<Pack> initial_k_field(field_mesh, "tank_initial_k");
    SimpleFluid::CellField<Pack> initial_omega_field(field_mesh, "tank_initial_omega");
    SimpleFluid::CellField<Pack> initial_nut_field(field_mesh, "tank_initial_nut");
    constexpr double sst_beta_1 = 0.075;
    constexpr double sst_cell_omega_coefficient = 6.0;
    const double molecular_kinematic_viscosity = dynamic_viscosity / density;
    for (size_t owned = 0; owned < field_mesh->num_owned_cells(); ++owned)
    {
        const auto lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const double distance = exact_wall_distance.value(lid);
        const double omega = std::max(initial_omega,
            sst_cell_omega_coefficient * molecular_kinematic_viscosity / (sst_beta_1 * distance * distance));
        initial_k_field.set_owned_value(lid, initial_k);
        initial_omega_field.set_owned_value(lid, omega);
        initial_nut_field.set_owned_value(lid, initial_k / omega);
    }
    initial_k_field.sync_ghosts();
    initial_omega_field.sync_ghosts();
    initial_nut_field.sync_ghosts();
    turbulence.restore_transported_state(initial_k_field, initial_omega_field, initial_nut_field, solver.velocity(),
        solver.material_properties(), density);

    const auto communicator = Tpetra::getDefaultComm();
    const int rank = communicator->getRank();
    long long local_cells = static_cast<long long>(mesh->num_owned_cells());
    long long global_cells = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
    const auto integrated_fission_power = fission.integrated_power();
    double minimum_wall_face_distance = 0.0;
    double maximum_wall_face_distance = 0.0;
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1, &local_minimum_wall_face_distance, &minimum_wall_face_distance);
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MAX, 1, &local_maximum_wall_face_distance, &maximum_wall_face_distance);
    if (rank == 0)
    {
        std::cout << "Gaussian tank SST: R=" << radius << " m, H=" << height << " m, nominal R/Z spacing=" << rz_spacing
                  << " m, circumferential spacing=" << circumferential_spacing << " m, cells=" << global_cells
                  << ", ranks=" << communicator->getSize() << '\n'
                  << "Boundary layer: count=" << boundary_layer_count
                  << ", first_cell_height=" << boundary_layer_first_height << " m, growth=" << boundary_layer_growth
                  << ", wall_distance=analytic cylinder"
                  << ", wall_face_normal_distance=[" << minimum_wall_face_distance << ',' << maximum_wall_face_distance
                  << "] m"
                  << ", omega_initialization=wall-compatible"
                  << ", axisymmetric_projection=yes\n"
                  << "Gaussian source: requested_power=" << total_power
                  << " W, integrated_power=" << integrated_fission_power << " W\n";
    }

    SimpleFluid::ProgressStream progress(std::cout);
    for (int step = 0; step < steps; ++step)
    {
        solver.step(progress);
        project_axisymmetric_tank_state(solver, turbulence, radial_projection_edges, axial_projection_edges, density);
        if ((step + 1) % 10 == 0 || step + 1 == steps)
        {
            const auto courant_number = solver.maximum_courant_number();
            if (rank == 0)
            {
                std::cout << "tank_diagnostics: step=" << step + 1 << ", time=" << solver.time()
                          << " s, maximum_Courant=" << courant_number << '\n';
            }
        }
    }
    const auto maximum_courant_number = solver.maximum_courant_number();

    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_sources = true;
    output_options.include_material_properties = true;
    output_options.include_turbulence_fields = true;
    solver.write_parallel_solution_vtu("fissile_solution_tank_sst.vtu", output_options);
    write_profile_cells(*mesh, solver, turbulence, fission, rank);

    if (rank == 0)
    {
        std::cout << "Gaussian tank SST complete: steps=" << solver.step_index() << ", t=" << solver.time()
                  << " s, maximum_Courant=" << maximum_courant_number
                  << ", integrated_power=" << integrated_fission_power << " W\n";
    }
    return 0;
}
