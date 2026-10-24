/**
 * @file coupling_kernel_benchmark.cc
 * @brief Opt-in gas kinetics and annular ALE checkpoint replay measurements.
 *
 * Run both builds with identical arguments, one thread per rank, and fresh output
 * directories. Gas repeats restore the same initial state outside the measured
 * advance; ALE measures checkpoint creation, all trials/replays, and acceptance.
 * CSV output and additional verification are outside the measured regions.
 * Modes retain fixed-R100 chemistry and annular coupling/replay conservation workloads;
 * the explicit finite two-gas fixture is versioned separately from retired H2 fallback timings.
 */
#include "../examples/FiniteGasVerificationSupport.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include <memory>

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace FG = SimpleFluid::Verification::FiniteGas;
namespace M = SimpleFluid::Examples::ReactingMixture;
namespace R = SimpleFluid::Reacting;
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Solver = SimpleFluid::BoussinesqSolver<Pack>;
using Field = SimpleFluid::ScalarCellFieldStored<Pack, Mesh>;
using Gas = SimpleFluid::ReactingGasModel<Pack, Mesh>;
using Local = Pack::local_ordinal_type;
using Clock = std::chrono::steady_clock;

struct Options
{
    std::string mode = "gas";
    std::filesystem::path output;
    int nx = 0, ny = 0, nz = 0, steps = 0, replays = 2;
};

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help")
            throw std::invalid_argument("usage: coupling_kernel_benchmark --mode gas|ale --output NEW_DIR "
                "[--nx N --ny N --nz N --steps N --replays N]; gas defaults: 8x8x8,20 advances; "
                "ALE defaults: 4x8x8,4 intervals,2 trials per interval; each axis >=2");
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + argument);
        const std::string value = argv[++i];
        if (argument == "--mode") options.mode = value;
        else if (argument == "--output") options.output = value;
        else if (argument == "--nx") options.nx = std::stoi(value);
        else if (argument == "--ny") options.ny = std::stoi(value);
        else if (argument == "--nz") options.nz = std::stoi(value);
        else if (argument == "--steps") options.steps = std::stoi(value);
        else if (argument == "--replays") options.replays = std::stoi(value);
        else throw std::invalid_argument("Unknown argument " + argument);
    }
    if (options.mode != "gas" && options.mode != "ale") throw std::invalid_argument("Invalid mode");
    const bool gas = options.mode == "gas";
    if (!options.nx) options.nx = gas ? 8 : 4;
    if (!options.ny) options.ny = 8;
    if (!options.nz) options.nz = 8;
    if (!options.steps) options.steps = gas ? 20 : 4;
    if (options.output.empty() || options.nx < 2 || options.ny < 2 || options.nz < 2 ||
        options.nx > 128 || options.ny > 128 || options.nz > 128 ||
        options.steps < 1 || options.steps > 1000 || options.replays < 1 || options.replays > 100)
        throw std::invalid_argument("Invalid output, grid (2..128), steps (1..1000), or replays (1..100)");
    return options;
}

double reduce(double local, Teuchos::EReductionType operation = Teuchos::REDUCE_SUM)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), operation, 1, &local, &result);
    return result;
}

void require(bool local, const std::string& message)
{
    if (reduce(local ? 1.0 : 0.0, Teuchos::REDUCE_MIN) == 0.0)
        throw std::runtime_error(message);
}

SimpleFluid::ArrReal edges(double lower, double upper, int cells)
{
    SimpleFluid::ArrReal values(static_cast<size_t>(cells) + 1);
    for (int i = 0; i <= cells; ++i) values[static_cast<size_t>(i)] = std::lerp(lower, upper, double(i) / cells);
    return values;
}

Mesh::DistributionOptions distribution()
{
    const auto comm = Tpetra::getDefaultComm();
    Mesh::DistributionOptions result;
    result.partition = static_cast<size_t>(comm->getRank());
    result.partitions = static_cast<size_t>(comm->getSize());
    result.allow_empty_partitions = true;
    return result;
}

std::ofstream output_file(const Options& options, const std::string& name)
{
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(options.output / (name + "-rank" + std::to_string(Tpetra::getDefaultComm()->getRank()) + ".csv"));
    stream << std::setprecision(17);
    return stream;
}

void write_fields(const Options& options, const Mesh& mesh,
    const std::map<std::string, const Field*>& fields,
    const SimpleFluid::VectorCellFieldStored<Pack, Mesh>* velocity = nullptr)
{
    bool finite = true;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        for (const auto& [name, field] : fields) finite = finite && std::isfinite(field->value(cell));
        if (velocity)
        {
            const auto value = velocity->value(cell);
            finite = finite && std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        }
    }
    require(finite, "Non-finite final field state");
    auto stream = output_file(options, "fields");
    stream << "mode,rank,gid,volume_m3,x_m,y_m,z_m";
    for (const auto& [name, field] : fields) stream << ',' << name;
    if (velocity) stream << ",ux_m_s,uy_m_s,uz_m_s";
    stream << '\n';
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto center = mesh.cell_centroid(cell);
        stream << options.mode << ',' << Tpetra::getDefaultComm()->getRank() << ',' << mesh.cell_geometry_global_id(cell)
               << ',' << mesh.cell_volume(cell) << ',' << center.x << ',' << center.y << ',' << center.z;
        for (const auto& [name, field] : fields) stream << ',' << field->value(cell);
        if (velocity)
        {
            const auto value = velocity->value(cell);
            stream << ',' << value.x << ',' << value.y << ',' << value.z;
        }
        stream << '\n';
    }
}

void write_summary(const Options& options, const Mesh& mesh, double wall, double cpu, int advances,
    const Gas& gas, double time, double mass = 0.0, double energy = 0.0, double top = 0.0)
{
    const double max_wall = reduce(wall, Teuchos::REDUCE_MAX);
    const double sum_cpu = reduce(cpu);
    const double inventory = FG::total(gas);
    double local_substeps = 0, mass_error = 0;
    for (const auto& ledger : gas.source_ledgers())
        local_substeps = std::max(local_substeps, static_cast<double>(ledger.substeps));
    const double maximum_substeps = reduce(local_substeps, Teuchos::REDUCE_MAX);
    for (const auto& h : gas.species_histories())
        mass_error += h.inventory_error * gas.options()
                                              .reaction_mechanism->catalog()
                                              .species()[gas.options().reaction_mechanism->catalog().index(h.species)]
                                              .molarMass;
    const double bubble_volume = gas.global_submerged_bubble_volume();
    auto stream = output_file(options, "summary");
    stream
        << "fixture_schema,mode,rank,ranks,nx,ny,nz,global_cells,steps,trials_per_interval,advances,time_s,wall_s,max_"
           "rank_wall_s,"
           "cpu_s,sum_rank_cpu_s,gas_mol,gas_source_mol,gas_outflow_mol,species_inventory_error_mol,mass_budget_error_"
           "kg,"
           "bubble_volume_m3,maximum_reaction_substeps,mass_kg,energy_J,top_m\n";
    stream << 2 << ',' << options.mode << ',' << Tpetra::getDefaultComm()->getRank() << ','
           << Tpetra::getDefaultComm()->getSize() << ',' << options.nx << ',' << options.ny << ',' << options.nz << ','
           << mesh.owned_cell_map()->getGlobalNumElements() << ',' << options.steps << ','
           << (options.mode == "ale" ? options.replays : 1) << ',' << advances << ',' << time << ',' << wall << ','
           << max_wall << ',' << cpu << ',' << sum_cpu << ',' << inventory << ',' << FG::produced(gas) << ','
           << FG::escaped(gas) << ',' << FG::inventory_error(gas) << ',' << mass_error << ',' << bubble_volume << ','
           << maximum_substeps << ',' << mass << ',' << energy << ',' << top << '\n';
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << std::setprecision(17) << options.mode << ": " << advances << " advances, max-rank wall "
                  << max_wall << " s, summed CPU " << sum_cpu << " s; output " << options.output << '\n';
}

void run_gas(const Options& options)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        edges(0, 1, options.nx), edges(0, 1, options.ny), edges(0, 1, options.nz)}});
    auto mesh = std::make_shared<Mesh>(geometry, distribution());
    constexpr double dt = 2e-6, reaction_rate = .01, reaction_frequency = 2.5e6;
    M::Mixture mixture(false, false, R::EnergyConvention::SensibleEnergy, 0);
    M::InitialMixture seed;
    seed.reactant_concentration = 0;
    seed.dissolved_one_concentration = seed.dissolved_two_concentration = 500;
    seed.populations = {{"micro", 1e8, 1e-16, .25}, {"large", 1e6, 1e-13, .25}};
    auto settings = M::field_options(mixture, seed, reaction_rate, 0);
    settings.species_properties = {mixture.transport(M::gas_one, 0, 0), mixture.transport(M::gas_two, 0, 0)};
    auto provider = mixture.reaction_rate(0, R::SourceVolumeBasis::BulkVolume);
    provider.identity = "manufactured.finite_pulse_stiff_mass_action";
    provider.parameterUnits = "reaction frequency 1/s; separately declared finite donor pulse mol/cell";
    provider.evaluate = [reaction_frequency](const R::ReactionContext& context)
    {
        const double amount = context.state.amount(context.catalog, M::reactant, R::SpeciesPhase::Liquid);
        return std::vector<R::ReactionRate>{{M::conversion,
            reaction_frequency * amount / context.environment.bulkVolume, R::SourceVolumeBasis::BulkVolume}};
    };
    settings.species_source = [provider, reaction_rate](const R::ReactionContext& context, double duration)
    {
        R::ReactionSourceInput input;
        input.rates = provider;
        const double amount = reaction_rate * context.environment.bulkVolume * duration;
        input.externalTransfers = {
            {"manufactured finite donor feed", M::reactant, R::SpeciesPhase::Liquid, amount, .010 * amount, 0, {}, {}}};
        return input;
    };
    settings.surface_tension = .07;
    settings.min_radius = 1e-12;
    settings.max_radius = 1e-3;
    settings.micro_dissolution_frequency = settings.large_dissolution_frequency = 1e7;
    settings.max_subcycles = 100;
    // This fixture keeps the R100 work oracle on the actual conservative SSPRK2 chemistry path.
    // A half-step margin makes the count robust to floating interval endpoint representation.
    settings.reaction_advance.maximumSubstep = dt / 99.5;
    settings.reaction_advance.maximumSubsteps = 100;
    Gas gas(mesh, settings);
    Field temperature(mesh, 300.0, "temperature"), pressure(mesh, 0.0, "pressure");
    SimpleFluid::VectorCellFieldStored<Pack, Mesh> velocity(mesh, Mesh::Vec3{}, "velocity");
    SimpleFluid::ScalarFaceFieldStored<Pack, Mesh> flux(mesh, 0.0, "flux");
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.reference_density = material_options.density = 1000.0;
    material_options.specific_heat_capacity = 4200.0;
    material_options.dynamic_viscosity = 1.e-3;
    material_options.thermal_conductivity = 0.6;
    SimpleFluid::MaterialPropertyFields<Pack, Mesh> material(mesh, material_options, SimpleFluid::TimeStepperOptions{});
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto center = mesh->cell_centroid(cell);
        temperature.set_owned_value(cell, 298.0 + 4.0 * center.z);
        material.density.set_owned_value(cell, 995.0 + 10.0 * center.x);
    }
    temperature.sync_ghosts();
    material.density.sync_ghosts();
    gas.initialize_state(0.0, temperature, pressure, velocity, material);
    const auto initial = gas.snapshot();
    double elapsed = 0.0, cpu = 0.0;
    // A warmup builds transport workspaces but is excluded from all measurements.
    gas.advance(dt, dt, temperature, pressure, velocity, flux, material);
    for (int step = 0; step < options.steps; ++step)
    {
        gas.restore(initial);
        Tpetra::getDefaultComm()->barrier();
        const auto start = Clock::now();
        const auto cpu_start = std::clock();
        gas.advance(dt, dt, temperature, pressure, velocity, flux, material);
        cpu += double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
        elapsed += std::chrono::duration<double>(Clock::now() - start).count();
        bool hundred = true;
        for (const auto& ledger : gas.source_ledgers())
            hundred = hundred && ledger.substeps == 100 && ledger.retries == 0;
        require(hundred, "Gas benchmark did not exercise 100 actual conservative chemistry substeps");
        for (const auto& h : gas.species_histories())
            require(std::isfinite(h.inventory_error) &&
                        std::abs(h.inventory_error) <= 1e-10 * std::max(1., std::abs(h.initial_moles)),
                "A finite species inventory closure failed");
        const double feed = reaction_rate * dt; // The physical fixed cube is exactly 1 m3.
        const auto amplification = [](long double z) { return 1 - z + .5L * z * z; };
        const double full_step = settings.reaction_advance.maximumSubstep;
        const double final_step = dt - 99 * full_step;
        const long double remaining =
            feed * std::pow(amplification(reaction_frequency * static_cast<long double>(full_step)), 99) *
            amplification(reaction_frequency * static_cast<long double>(final_step));
        const double extent = static_cast<double>(static_cast<long double>(feed) - remaining);
        for (const auto& h : gas.species_histories())
        {
            const double production =
                h.species == M::reactant
                    ? static_cast<double>(remaining)
                    : (h.species == M::product || h.species == M::gas_one || h.species == M::gas_two ? extent : 0.);
            if (h.species == M::reactant)
            {
                const auto endpoint_scale = std::max(remaining, static_cast<long double>(h.current_moles));
                const auto endpoint_bound = 4096 * std::numeric_limits<double>::epsilon() * endpoint_scale;
                require(h.current_moles > 0 && h.current_moles < feed * .01 &&
                            std::abs(static_cast<long double>(h.current_moles) - remaining) <= endpoint_bound &&
                            std::abs(h.external_supply - feed) <= 1e-18,
                    "The stiff donor must retain its discrete SSPRK2 endpoint and actual predecessor feed");
            }
            require(std::abs(h.current_moles - h.initial_moles - production) <=
                        1e-10 * std::max(1., std::abs(h.initial_moles)),
                "Independent manufactured reaction stoichiometry failed");
            if (h.species == M::gas_one || h.species == M::gas_two)
                require(std::abs(h.chemical_production - extent) <= 1e-10,
                    "Each manufactured gas must have its own actual chemical source");
        }
    }
    auto fields = FG::fields(gas);
    fields.emplace("temperature_K", &temperature);
    fields.emplace("density_kg_m3", &material.density);
    write_fields(options, *mesh, fields);
    write_summary(options, *mesh, elapsed, cpu, options.steps, gas, dt);
}

void run_ale(const Options& options)
{
    constexpr double inner = 0.03815, outer = 0.25, bottom = 0.1, initial_top = 0.60852, vessel_top = 2.1;
    constexpr double initial_mass = 151.09200565656749, initial_temperature = 298.85;
    constexpr double interval_dt = 0.01, interval_energy = 0.01, yield = 1.5e-7;
    const double area = std::numbers::pi * (outer * outer - inner * inner);
    const double density = initial_mass / (area * (initial_top - bottom));
    const double mass_tolerance = 4096 * std::numeric_limits<double>::epsilon() * initial_mass;
    auto geometry = std::make_shared<Mesh::Cylindrical>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        edges(inner, outer, options.nx), edges(0, 2 * std::numbers::pi, options.ny),
        edges(bottom, initial_top, options.nz)}});
    auto mesh = std::make_shared<Mesh>(geometry, distribution());
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"rmin", "rmax", "zmin", "zmax"})
    {
        boundaries.temperature[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.pressure[name] = {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    boundaries.pressure["zmax"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    boundaries.velocity["zmax"] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    SimpleFluid::TimeStepperOptions time;
    time.time_step = 0.005;
    time.steps = 1;
    time.reference_temperature = initial_temperature;
    time.thermal_expansion = 2.1e-4;
    time.gravity_z = -9.81;
    time.pressure_velocity_coupling = SimpleFluid::PressureVelocityCoupling::PISO;
    time.n_pressure_correctors = 2;
    SimpleFluid::LinearSolverOptions linear;
    linear.tolerance = 1.e-11;
    linear.max_iterations = 1000;
    SimpleFluid::BoussinesqModelOptions material;
    material.reference_density = material.density = density;
    material.specific_heat_capacity = 4200.0;
    material.dynamic_viscosity = 1.e-3;
    material.thermal_conductivity = 0.6;
    auto solver_shared = std::make_shared<Solver>(mesh, boundaries, time, linear, material);
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::MaterialFeedbackOptions thermal;
    thermal.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    thermal.reference_density = thermal.liquid_density = density;
    thermal.reference_temperature = initial_temperature;
    thermal.thermal_expansion = 2.1e-4;
    thermal.reference_dynamic_viscosity = 1.e-3;
    solver_coupling.configure_material_feedback(thermal);
    auto* feed = &solver_coupling.add_prescribed_heat_source();
    feed->initialize_constant(0.0);
    FG::Parameters reference{initial_temperature, 101325., density, 4200., 2.1e-4, 1e-3, .6, .07, 0., 0., 0., 0., 0.,
        1e-5, 50e-6, .01, 4.5e-9, 1e-5};
    auto gas_options = FG::options(
        FG::mixture(reference), reference,
        [feed, yield](const R::ReactionContext& context)
        {
            if (!context.environment.location || !context.environment.location->ownedCell)
                throw std::invalid_argument("A material-following gas feed requires its actual source frame.");
            return yield * feed->field().value(static_cast<Local>(*context.environment.location->ownedCell));
        },
        [](const R::ReactionContext& c) { return c.environment.temperature; });
    gas_options.transport_solver_tolerance = 1e-14;
    auto& gas = solver_coupling.configure_reacting_gas(gas_options);
    solver_coupling.initialize_linear_temperature({0.0, 0.0, 1.0}, initial_temperature, initial_temperature);
    SimpleFluid::FreeSurfaceOptions surface;
    surface.enabled = true;
    surface.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
    surface.gravity_axis = SimpleFluid::Dimension::Z;
    surface.range_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.initial_liquid_volume = area * (initial_top - bottom);
    surface.vessel.mode = SimpleFluid::VesselVolumeMapMode::ConstantArea;
    surface.vessel.bottom_elevation = bottom;
    surface.vessel.top_elevation = vessel_top;
    surface.vessel.cross_section_area = area;
    surface.vessel.total_internal_volume = area * (vessel_top - bottom);
    surface.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    surface.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    surface.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    surface.headspace.ambient_pressure = surface.headspace.initial_pressure = 101325.0;
    surface.headspace.initial_temperature = initial_temperature;
    surface.ale.top_boundary = "zmax";
    surface.ale.maximum_correctors = 30;
    require(solver_coupling.configure_free_surface(surface) != nullptr, "ALE setup failed");
    const std::array<double, 8> axial_shape{0.5, 0.8, 1.2, 1.5, 1.4, 1.1, 0.8, 0.5};
    const std::array<double, 4> radial_shape{1.0, 1.2, 0.9, 0.7};
    std::vector<double> energy(mesh->num_owned_cells());
    double local_weight = 0.0;
    for (size_t owned = 0; owned < energy.size(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        const auto gid = static_cast<size_t>(mesh->cell_geometry_global_id(cell));
        const auto radial = (gid % options.nx) * radial_shape.size() / options.nx;
        const auto axial = (gid / (options.nx * options.ny)) * axial_shape.size() / options.nz;
        energy[owned] = mesh->cell_volume(cell) * radial_shape[radial] * axial_shape[axial];
        local_weight += energy[owned];
    }
    const auto total_weight = reduce(local_weight);
    for (auto& value : energy) value *= interval_energy / total_weight;
    auto history = output_file(options, "history");
    history << "interval,trial,time_s,mass_kg,gas_source_mol,gas_mol,gas_outflow_mol,top_m,gcl_m3_s,continuity_1_s\n";
    const auto top = [&]
    {
        double result = bottom;
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
            if (mesh->boundary_batch_name(batch_id) == "zmax")
                for (const auto face : batch.face_lids)
                    if (mesh->is_owned_face(face)) result = std::max(result, mesh->face_centroid(face).z);
        return reduce(result, Teuchos::REDUCE_MAX);
    };
    double elapsed = 0.0, cpu = 0.0;
    int advances = 0;
    for (int interval = 0; interval < options.steps; ++interval)
    {
        const double start = interval * interval_dt, end = (interval + 1) * interval_dt, duration = end - start;
        const int substeps = static_cast<int>(std::ceil(duration / time.time_step));
        Tpetra::getDefaultComm()->barrier();
        auto begin = Clock::now();
        auto cpu_begin = std::clock();
        auto checkpoint = solver_coupling.create_coupling_checkpoint();
        elapsed += std::chrono::duration<double>(Clock::now() - begin).count();
        cpu += double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
        std::vector<double> first_temperature;
        std::vector<Mesh::Vec3> first_velocity;
        std::map<std::string, std::vector<double>> first_species_state;
        double first_top = 0.0;
        for (int trial = 0; trial < options.replays; ++trial)
        {
            Tpetra::getDefaultComm()->barrier();
            begin = Clock::now();
            cpu_begin = std::clock();
            solver_coupling.restore_coupling_checkpoint(checkpoint);
            solver_coupling.set_coupling_interval_energy(energy, duration);
            for (int substep = 0; substep < substeps; ++substep)
            {
                const double endpoint = substep + 1 == substeps ? end : start + duration * (substep + 1) / substeps;
                solver.set_time_step(endpoint - solver.time());
                solver_coupling.step();
                ++advances;
            }
            elapsed += std::chrono::duration<double>(Clock::now() - begin).count();
            cpu += double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
            const double mass = solver_coupling.liquid_mass_inventory().totalMass();
            const double produced = (interval + 1) * interval_energy * yield;
            const double retained = FG::total(gas);
            const double escaped = FG::escaped(gas);
            const double height = top();
            // The finite solution owns actual dissolved gas mass. No zero-mass solute approximation.
            const double expected_liquid_mass =
                initial_mass + FG::molar_feed_mass * produced - FG::gas_mass(gas) - FG::escaped_mass(gas);
            require(std::abs(mass - expected_liquid_mass) <= mass_tolerance && solver.time() == end &&
                        std::abs(FG::produced(gas) - produced) <= 1e-18 &&
                        std::abs(retained + escaped - produced) <= 1e-17,
                "ALE time/finite-mass/two-gas closure failed");
            for (const auto& h : gas.species_histories())
                if (std::ranges::find(FG::gas_ids, h.species) != FG::gas_ids.end())
                {
                    const double fraction = h.species == FG::gas_ids[0] ? FG::one_fraction : 1 - FG::one_fraction;
                    require(std::abs(h.external_supply - fraction * produced) <= 1e-18 &&
                                std::abs(h.current_moles + h.physical_outflow - fraction * produced) <= 1e-17,
                        "Independent gas source/escape budget failed");
                }
            bool replay_matches = true;
            // These are actual schema fields and each bin's count, not the retired five-field mirror.
            auto replay_fields = FG::fields(gas);
            replay_fields.emplace("mechanical_pressure_Pa", &solver.pressure());
            replay_fields.emplace("material_density_kg_m3", &solver.material_properties().density);
            for (const auto& [name, field] : replay_fields)
            {
                auto& baseline = first_species_state[name];
                for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
                {
                    const double value = field->value(static_cast<Local>(owned));
                    if (trial == 0)
                        baseline.push_back(value);
                    else
                        replay_matches =
                            replay_matches && std::abs(value - baseline.at(owned)) <=
                                                  1e-18 + 1e-10 * std::max(std::abs(value), std::abs(baseline[owned]));
                }
            }
            for (size_t owned = 0; owned < energy.size(); ++owned)
            {
                const auto cell = static_cast<Local>(owned);
                const double temperature = solver.temperature().value(cell);
                const auto velocity = solver.velocity().value(cell);
                if (trial == 0)
                {
                    first_temperature.push_back(temperature);
                    first_velocity.push_back(velocity);
                }
                else
                {
                    replay_matches = replay_matches && std::abs(temperature - first_temperature[owned]) <= 1e-11;
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        replay_matches = replay_matches && std::abs(velocity.component(axis) -
                                                                    first_velocity[owned].component(axis)) <= 1e-11;
                }
            }
            if (trial == 0) first_top = height;
            require(replay_matches && std::abs(height - first_top) <= 1.e-13, "ALE replay field mismatch");
            const auto& diagnostics = solver_coupling.planar_ale_diagnostics();
            history << interval << ',' << trial << ',' << solver.time() << ',' << mass << ',' << FG::produced(gas)
                    << ',' << retained << ',' << escaped << ',' << height << ',' << diagnostics.maximum_gcl_residual
                    << ',' << diagnostics.continuity.maximum << '\n';
        }
        begin = Clock::now();
        cpu_begin = std::clock();
        solver_coupling.accept_coupling_checkpoint(checkpoint);
        elapsed += std::chrono::duration<double>(Clock::now() - begin).count();
        cpu += double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
    }
    auto fields = FG::fields(gas);
    fields.emplace("temperature_K", &solver.temperature());
    fields.emplace("pressure_Pa", &solver.pressure());
    fields.emplace("density_kg_m3", &solver.material_properties().density);
    fields.emplace("liquid_inventory_kg_m3", &solver_coupling.liquid_mass_inventory().cellMassInventory());
    fields.emplace("power_W_m3", &solver_coupling.find_prescribed_heat_source()->field());
    write_fields(options, *mesh, fields, &solver.velocity());
    long double local_energy = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Local>(owned);
        local_energy +=
            static_cast<long double>(solver_coupling.liquid_mass_inventory().cellMassInventory().value(cell)) *
            mesh->cell_volume(cell) * material.specific_heat_capacity * solver.temperature().value(cell);
    }
    const double total_energy = reduce(static_cast<double>(local_energy));
    const double mass = solver_coupling.liquid_mass_inventory().totalMass();
    const double height = top();
    write_summary(options, *mesh, elapsed, cpu, advances, gas, solver.time(), mass, total_energy, height);
}
} // namespace

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    try
    {
        const auto options = parse(argc, argv);
        bool created = true;
        if (Tpetra::getDefaultComm()->getRank() == 0)
        {
            std::error_code error;
            created = std::filesystem::create_directories(options.output, error) && !error;
        }
        require(created, "Output directory must be new and writable");
        Tpetra::getDefaultComm()->barrier();
        if (options.mode == "gas") run_gas(options);
        else run_ale(options);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "coupling_kernel_benchmark rank " << Tpetra::getDefaultComm()->getRank() << ": " << error.what() << '\n';
        return 1;
    }
}
