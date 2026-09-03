/**
 * @file PlanarFreeSurfaceModel.cc
 * @brief Fixed-grid planar liquid-level and ideal-gas headspace models.
 */

#include "PlanarFreeSurfaceModel.hh"

#include "dataclass/DatabaseOptionReader.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace SimpleFluid
{
namespace
{

[[nodiscard]] bool finite_positive(real_t value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] real_t sum_moles(const GasMolesBySpecies& moles)
{
    real_t total = 0.0;
    for (const auto& [species, value] : moles)
    {
        if (species.empty())
        {
            throw std::invalid_argument("Gas species names cannot be empty.");
        }
        if (!std::isfinite(value) || value < 0.0)
        {
            throw std::invalid_argument("Gas moles for species '" + species + "' must be finite and non-negative.");
        }
        total += value;
    }
    if (!std::isfinite(total))
    {
        throw std::overflow_error("Total gas moles are not finite.");
    }
    return total;
}

void add_moles(GasMolesBySpecies& destination, const GasMolesBySpecies& source)
{
    static_cast<void>(sum_moles(source));
    for (const auto& [species, value] : source)
    {
        destination[species] += value;
        if (!std::isfinite(destination[species]))
        {
            throw std::overflow_error("Accumulated gas moles for species '" + species + "' are not finite.");
        }
    }
}

[[nodiscard]] GasMolesBySpecies added_moles(GasMolesBySpecies destination, const GasMolesBySpecies& source)
{
    add_moles(destination, source);
    return destination;
}

[[nodiscard]] std::string range_error(std::string_view quantity, real_t requested, real_t lower, real_t upper)
{
    std::ostringstream message;
    message << "Vessel " << quantity << " " << requested << " is outside the supported range [" << lower << ", "
            << upper << "].";
    return message.str();
}

[[nodiscard]] FreeSurfaceRangePolicy parse_range_policy(const std::string& value)
{
    if (value == "error")
    {
        return FreeSurfaceRangePolicy::Error;
    }
    if (value == "clampAndReport")
    {
        return FreeSurfaceRangePolicy::ClampAndReport;
    }
    throw std::invalid_argument("free_surface_overflow_policy must be 'error' or 'clampAndReport'.");
}

[[nodiscard]] FreeSurfaceMode parse_free_surface_mode(const std::string& value)
{
    if (value == "fixed")
    {
        return FreeSurfaceMode::Fixed;
    }
    if (value == "planarVolumeBudget")
    {
        return FreeSurfaceMode::PlanarVolumeBudget;
    }
    if (value == "planarALE")
    {
        return FreeSurfaceMode::PlanarALE;
    }
    throw std::invalid_argument("free_surface_model must be 'fixed', 'planarVolumeBudget', or 'planarALE'.");
}

[[nodiscard]] VesselVolumeMapMode parse_vessel_mode(const std::string& value)
{
    if (value == "constantArea")
    {
        return VesselVolumeMapMode::ConstantArea;
    }
    if (value == "tabulated")
    {
        return VesselVolumeMapMode::Tabulated;
    }
    throw std::invalid_argument("free_surface_vessel_model must be 'constantArea' or 'tabulated'.");
}

[[nodiscard]] LiquidVolumeMode parse_liquid_volume_mode(const std::string& value)
{
    if (value == "globalConstantMass")
    {
        return LiquidVolumeMode::GlobalConstantMass;
    }
    if (value == "cellMassInventory")
    {
        return LiquidVolumeMode::CellMassInventory;
    }
    throw std::invalid_argument(
        "free_surface_liquid_volume_model must be 'globalConstantMass' or 'cellMassInventory'.");
}

[[nodiscard]] HeadspaceMode parse_headspace_mode(const std::string& value)
{
    if (value == "vented")
    {
        return HeadspaceMode::Vented;
    }
    if (value == "closed")
    {
        return HeadspaceMode::Closed;
    }
    if (value == "restrictedVent")
    {
        throw std::invalid_argument("free_surface_headspace_model 'restrictedVent' is reserved but not implemented "
                                    "because no validated vent law is available.");
    }
    throw std::invalid_argument("free_surface_headspace_model must be 'vented' or 'closed'.");
}

[[nodiscard]] HeadspaceTemperatureMode parse_temperature_mode(const std::string& value)
{
    if (value == "fixed")
    {
        return HeadspaceTemperatureMode::Fixed;
    }
    if (value == "prescribed")
    {
        return HeadspaceTemperatureMode::Prescribed;
    }
    if (value == "bulkLiquid")
    {
        return HeadspaceTemperatureMode::BulkLiquid;
    }
    throw std::invalid_argument(
        "free_surface_headspace_temperature_model must be 'fixed', 'prescribed', or 'bulkLiquid'.");
}

[[nodiscard]] Dimension parse_gravity_axis(const std::string& value)
{
    if (value == "x")
    {
        return Dimension::X;
    }
    if (value == "y")
    {
        return Dimension::Y;
    }
    if (value == "z")
    {
        return Dimension::Z;
    }
    throw std::invalid_argument("free_surface_gravity_axis must be 'x', 'y', or 'z'.");
}

[[nodiscard]] real_t configured_usable_volume(const FreeSurfaceVesselOptions& vessel)
{
    if (vessel.mode == VesselVolumeMapMode::ConstantArea)
    {
        return vessel.cross_section_area * (vessel.top_elevation - vessel.bottom_elevation);
    }
    if (vessel.volume_table.empty())
    {
        return std::numeric_limits<real_t>::quiet_NaN();
    }
    return vessel.volume_table.back();
}

void validate_coupling_options(const FreeSurfaceCouplingOptions& coupling)
{
    if (coupling.maximum_correctors <= 0)
    {
        throw std::invalid_argument("free_surface_coupling_max_correctors must be positive.");
    }
    if (!std::isfinite(coupling.absolute_tolerance) || coupling.absolute_tolerance <= 0.0 ||
        !std::isfinite(coupling.relative_tolerance) || coupling.relative_tolerance < 0.0)
    {
        throw std::invalid_argument("Free-surface coupling tolerances must be finite; the absolute tolerance must be "
                                    "positive and the relative tolerance non-negative.");
    }
    if (!std::isfinite(coupling.relaxation) || coupling.relaxation <= 0.0 || coupling.relaxation > 1.0)
    {
        throw std::invalid_argument("free_surface_coupling_relaxation must be in (0, 1].");
    }
    if (!finite_positive(coupling.minimum_absolute_pressure) || !finite_positive(coupling.maximum_absolute_pressure) ||
        coupling.minimum_absolute_pressure >= coupling.maximum_absolute_pressure)
    {
        throw std::invalid_argument(
            "Free-surface absolute-pressure bounds must be finite, positive, and strictly increasing.");
    }
    if (!std::isfinite(coupling.volume_absolute_tolerance) || coupling.volume_absolute_tolerance < 0.0 ||
        !std::isfinite(coupling.volume_relative_tolerance) || coupling.volume_relative_tolerance < 0.0 ||
        !std::isfinite(coupling.gas_absolute_tolerance) || coupling.gas_absolute_tolerance < 0.0 ||
        !std::isfinite(coupling.gas_relative_tolerance) || coupling.gas_relative_tolerance < 0.0)
    {
        throw std::invalid_argument("Free-surface volume and gas closure tolerances must be finite and non-negative.");
    }
}

void validate_headspace_options(const HeadspaceOptions& headspace)
{
    if (!finite_positive(headspace.ambient_pressure) || !finite_positive(headspace.initial_pressure))
    {
        throw std::invalid_argument("Headspace pressures must be finite positive absolute pressures [Pa].");
    }
    if (!finite_positive(headspace.initial_temperature))
    {
        throw std::invalid_argument("Headspace initial temperature must be finite and positive [K].");
    }
    if (!finite_positive(headspace.gas_constant) || !finite_positive(headspace.compressibility_factor))
    {
        throw std::invalid_argument("Headspace gas constant and compressibility factor must be finite and positive.");
    }
    if (!finite_positive(headspace.total_internal_volume))
    {
        throw std::invalid_argument("Headspace total internal vessel volume must be finite and positive [m^3].");
    }
    const auto initial_moles = sum_moles(headspace.initial_moles);
    if (headspace.mode == HeadspaceMode::Vented)
    {
        if (!headspace.initial_moles.empty() || !headspace.infer_initial_moles)
        {
            throw std::invalid_argument(
                "Explicit initial headspace moles are supported only by the closed headspace model.");
        }
    }
    else if (headspace.infer_initial_moles && !headspace.initial_moles.empty())
    {
        throw std::invalid_argument(
            "Closed-headspace initial moles are ambiguous: either infer them from the initial state or provide an "
            "explicit inventory, but not both.");
    }
    else if (!headspace.infer_initial_moles && !(initial_moles > 0.0))
    {
        throw std::invalid_argument("A closed headspace with explicit initial moles requires a positive inventory.");
    }
    if (headspace.temperature_mode == HeadspaceTemperatureMode::Prescribed)
    {
        if (headspace.prescribed_temperature_times.empty() ||
            headspace.prescribed_temperature_times.size() != headspace.prescribed_temperature_values.size())
        {
            throw std::invalid_argument(
                "Prescribed headspace temperature times and values must have the same nonzero size.");
        }
        for (size_t index = 0; index < headspace.prescribed_temperature_times.size(); ++index)
        {
            if (!std::isfinite(headspace.prescribed_temperature_times[index]) ||
                !finite_positive(headspace.prescribed_temperature_values[index]))
            {
                throw std::invalid_argument(
                    "Prescribed headspace times must be finite and temperatures finite and positive.");
            }
            if (index > 0 &&
                !(headspace.prescribed_temperature_times[index] > headspace.prescribed_temperature_times[index - 1]))
            {
                throw std::invalid_argument("Prescribed headspace temperature times must be strictly increasing.");
            }
        }
    }
}

void validate_gas_inventory(const FreeSurfaceGasInventory& gas)
{
    static_cast<void>(sum_moles(gas.initial_moles));
    static_cast<void>(sum_moles(gas.generated_moles));
    static_cast<void>(sum_moles(gas.dissolved_moles));
    static_cast<void>(sum_moles(gas.submerged_moles));
    for (const auto& [population, moles] : gas.submerged_population_moles)
    {
        if (population.empty())
        {
            throw std::invalid_argument("Submerged gas population names must be non-empty.");
        }
        static_cast<void>(sum_moles(moles));
    }
    static_cast<void>(sum_moles(gas.escaped_moles_this_step));
    static_cast<void>(sum_moles(gas.other_sink_moles));
}

[[nodiscard]] real_t checked_volume(
    const FreeSurfaceUpdate::VolumeAtPressure& callback, real_t pressure, std::string_view label, bool permit_negative)
{
    real_t value = 0.0;
    try
    {
        value = callback(pressure);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(std::string(label) + " callback failed at absolute pressure " +
                                 std::to_string(pressure) + " Pa: " + error.what());
    }
    if (!std::isfinite(value) || (!permit_negative && value < 0.0))
    {
        throw std::invalid_argument(std::string(label) + " callback must return a finite non-negative volume [m^3].");
    }
    return value;
}

[[nodiscard]] GasMolesBySpecies gas_closure(const GasMolesBySpecies& initial, const FreeSurfaceGasInventory& gas,
    const GasMolesBySpecies& headspace, const GasMolesBySpecies& vented)
{
    std::set<std::string> species;
    auto insert_species = [&species](const GasMolesBySpecies& values)
    {
        for (const auto& [name, value] : values)
        {
            static_cast<void>(value);
            species.insert(name);
        }
    };
    insert_species(initial);
    insert_species(gas.generated_moles);
    insert_species(gas.dissolved_moles);
    insert_species(gas.submerged_moles);
    insert_species(headspace);
    insert_species(vented);
    insert_species(gas.other_sink_moles);

    auto value_or_zero = [](const GasMolesBySpecies& values, const std::string& name)
    {
        const auto iterator = values.find(name);
        return iterator == values.end() ? 0.0 : iterator->second;
    };
    GasMolesBySpecies closure;
    for (const auto& name : species)
    {
        closure[name] = value_or_zero(initial, name) + value_or_zero(gas.generated_moles, name) -
                        value_or_zero(gas.dissolved_moles, name) - value_or_zero(gas.submerged_moles, name) -
                        value_or_zero(headspace, name) - value_or_zero(vented, name) -
                        value_or_zero(gas.other_sink_moles, name);
    }
    return closure;
}

} // namespace

VesselVolumeMap::VesselVolumeMap(FreeSurfaceRangePolicy range_policy) noexcept : d_range_policy(range_policy) {}

VesselRangeEvaluation VesselVolumeMap::boundAndEvaluate(real_t requested, real_t lower, real_t upper,
    std::string_view quantity, const std::function<real_t(real_t)>& evaluator) const
{
    if (!std::isfinite(requested))
    {
        throw std::invalid_argument("Vessel " + std::string(quantity) + " must be finite.");
    }
    VesselRangeEvaluation result;
    result.requested = requested;
    result.accepted = requested;
    if (requested < lower)
    {
        if (d_range_policy == FreeSurfaceRangePolicy::Error)
        {
            throw std::out_of_range(range_error(quantity, requested, lower, upper));
        }
        result.accepted = lower;
        result.underflow = lower - requested;
    }
    else if (requested > upper)
    {
        if (d_range_policy == FreeSurfaceRangePolicy::Error)
        {
            throw std::out_of_range(range_error(quantity, requested, lower, upper));
        }
        result.accepted = upper;
        result.overflow = requested - upper;
    }
    result.value = evaluator(result.accepted);
    if (!std::isfinite(result.value))
    {
        throw std::runtime_error("Vessel map produced a non-finite " + std::string(quantity) + " result.");
    }
    d_last_range_evaluation = result;
    return result;
}

VesselRangeEvaluation VesselVolumeMap::evaluateVolumeBelow(real_t height) const
{
    return boundAndEvaluate(height, bottomElevation(), topElevation(), "height",
        [this](real_t bounded_height) { return volumeBelowInRange(bounded_height); });
}

VesselRangeEvaluation VesselVolumeMap::evaluateLevelForVolume(real_t volume) const
{
    return boundAndEvaluate(volume, 0.0, totalUsableVolume(), "volume",
        [this](real_t bounded_volume) { return levelForVolumeInRange(bounded_volume); });
}

real_t VesselVolumeMap::volumeBelow(real_t height) const
{
    return evaluateVolumeBelow(height).value;
}

real_t VesselVolumeMap::areaAt(real_t height) const
{
    return boundAndEvaluate(height, bottomElevation(), topElevation(), "height for area",
        [this](real_t bounded_height) { return areaAtInRange(bounded_height); })
        .value;
}

real_t VesselVolumeMap::levelForVolume(real_t volume) const
{
    return evaluateLevelForVolume(volume).value;
}

FreeSurfaceRangePolicy VesselVolumeMap::rangePolicy() const noexcept
{
    return d_range_policy;
}

const VesselRangeEvaluation& VesselVolumeMap::lastRangeEvaluation() const noexcept
{
    return d_last_range_evaluation;
}

ConstantAreaVesselVolumeMap::ConstantAreaVesselVolumeMap(
    real_t bottom_elevation, real_t top_elevation, real_t cross_section_area, FreeSurfaceRangePolicy range_policy)
    : VesselVolumeMap(range_policy), d_bottom_elevation(bottom_elevation), d_top_elevation(top_elevation),
      d_cross_section_area(cross_section_area)
{
    if (!std::isfinite(d_bottom_elevation) || !std::isfinite(d_top_elevation) ||
        !(d_top_elevation > d_bottom_elevation))
    {
        throw std::invalid_argument("Constant-area vessel elevations must be finite and strictly increasing.");
    }
    if (!finite_positive(d_cross_section_area))
    {
        throw std::invalid_argument("Constant-area vessel cross-section area must be finite and positive.");
    }
    if (!std::isfinite(totalUsableVolume()))
    {
        throw std::overflow_error("Constant-area vessel usable volume is not finite.");
    }
}

real_t ConstantAreaVesselVolumeMap::bottomElevation() const noexcept
{
    return d_bottom_elevation;
}

real_t ConstantAreaVesselVolumeMap::topElevation() const noexcept
{
    return d_top_elevation;
}

real_t ConstantAreaVesselVolumeMap::totalUsableVolume() const noexcept
{
    return d_cross_section_area * (d_top_elevation - d_bottom_elevation);
}

real_t ConstantAreaVesselVolumeMap::volumeBelowInRange(real_t height) const
{
    return d_cross_section_area * (height - d_bottom_elevation);
}

real_t ConstantAreaVesselVolumeMap::areaAtInRange(real_t) const
{
    return d_cross_section_area;
}

real_t ConstantAreaVesselVolumeMap::levelForVolumeInRange(real_t volume) const
{
    return d_bottom_elevation + volume / d_cross_section_area;
}

TabulatedVesselVolumeMap::TabulatedVesselVolumeMap(
    ArrReal heights, ArrReal volumes, FreeSurfaceRangePolicy range_policy)
    : VesselVolumeMap(range_policy), d_heights(std::move(heights)), d_volumes(std::move(volumes))
{
    if (d_heights.size() < 2 || d_heights.size() != d_volumes.size())
    {
        throw std::invalid_argument(
            "Tabulated vessel height and volume tables must have the same size of at least two.");
    }
    for (size_t index = 0; index < d_heights.size(); ++index)
    {
        if (!std::isfinite(d_heights[index]) || !std::isfinite(d_volumes[index]))
        {
            throw std::invalid_argument("Tabulated vessel height and volume values must be finite.");
        }
        if (index > 0 && !(d_heights[index] > d_heights[index - 1]))
        {
            throw std::invalid_argument("Tabulated vessel heights must be strictly increasing.");
        }
        if (index > 0 && d_volumes[index] < d_volumes[index - 1])
        {
            throw std::invalid_argument("Tabulated vessel volumes must be nondecreasing.");
        }
    }
    if (d_volumes.front() != 0.0)
    {
        throw std::invalid_argument("Tabulated vessel volume at the bottom elevation must be zero.");
    }
    if (!(d_volumes.back() > 0.0))
    {
        throw std::invalid_argument("Tabulated vessel usable volume must be positive.");
    }
}

real_t TabulatedVesselVolumeMap::bottomElevation() const noexcept
{
    return d_heights.front();
}

real_t TabulatedVesselVolumeMap::topElevation() const noexcept
{
    return d_heights.back();
}

real_t TabulatedVesselVolumeMap::totalUsableVolume() const noexcept
{
    return d_volumes.back();
}

size_t TabulatedVesselVolumeMap::heightSegment(real_t height) const
{
    if (height >= d_heights.back())
    {
        return d_heights.size() - 2;
    }
    const auto upper = std::upper_bound(d_heights.begin(), d_heights.end(), height);
    return static_cast<size_t>(std::distance(d_heights.begin(), upper) - 1);
}

size_t TabulatedVesselVolumeMap::volumeSegment(real_t volume) const
{
    if (volume >= d_volumes.back())
    {
        return d_volumes.size() - 2;
    }
    const auto upper = std::upper_bound(d_volumes.begin(), d_volumes.end(), volume);
    return static_cast<size_t>(std::distance(d_volumes.begin(), upper) - 1);
}

real_t TabulatedVesselVolumeMap::volumeBelowInRange(real_t height) const
{
    if (height == d_heights.front())
    {
        return d_volumes.front();
    }
    if (height == d_heights.back())
    {
        return d_volumes.back();
    }
    const auto lower = heightSegment(height);
    const auto fraction = (height - d_heights[lower]) / (d_heights[lower + 1] - d_heights[lower]);
    return d_volumes[lower] + fraction * (d_volumes[lower + 1] - d_volumes[lower]);
}

real_t TabulatedVesselVolumeMap::areaAtInRange(real_t height) const
{
    const auto lower = heightSegment(height);
    const auto area = (d_volumes[lower + 1] - d_volumes[lower]) / (d_heights[lower + 1] - d_heights[lower]);
    if (!(area > 0.0) || !std::isfinite(area))
    {
        throw std::domain_error(
            "Tabulated vessel has zero usable cross-section area at height " + std::to_string(height) + ".");
    }
    return area;
}

real_t TabulatedVesselVolumeMap::levelForVolumeInRange(real_t volume) const
{
    const auto equal = std::equal_range(d_volumes.begin(), d_volumes.end(), volume);
    if (equal.first != equal.second)
    {
        if (std::distance(equal.first, equal.second) > 1)
        {
            throw std::domain_error("Tabulated vessel inverse is non-unique at a zero-area volume plateau.");
        }
        return d_heights[static_cast<size_t>(std::distance(d_volumes.begin(), equal.first))];
    }

    const auto lower = volumeSegment(volume);
    const auto delta_volume = d_volumes[lower + 1] - d_volumes[lower];
    if (!(delta_volume > 0.0))
    {
        throw std::domain_error("Tabulated vessel inverse encountered a zero-area volume plateau.");
    }
    const auto fraction = (volume - d_volumes[lower]) / delta_volume;
    return d_heights[lower] + fraction * (d_heights[lower + 1] - d_heights[lower]);
}

FreeSurfaceOptions free_surface_options_from_database(const Database& database)
{
    const detail::DatabaseOptionReader reader(database, "Planar free-surface model");
    FreeSurfaceOptions options;
    options.enabled = reader.value_or<bool>("free_surface_enabled", options.enabled);
    options.mode = parse_free_surface_mode(reader.value_or<std::string>("free_surface_model", "fixed"));
    options.gravity_axis = parse_gravity_axis(reader.value_or<std::string>("free_surface_gravity_axis", "z"));
    options.validity_warning_relative_level_change = reader.value_or<real_t>(
        "free_surface_validity_warning_relative_level_change", options.validity_warning_relative_level_change);
    options.range_policy = parse_range_policy(reader.value_or<std::string>("free_surface_overflow_policy", "error"));
    if (reader.contains("free_surface_initial_liquid_volume"))
    {
        options.initial_liquid_volume = reader.required<real_t>("free_surface_initial_liquid_volume");
    }
    if (reader.contains("free_surface_initial_clear_level"))
    {
        options.initial_clear_level = reader.required<real_t>("free_surface_initial_clear_level");
    }

    options.vessel.mode = parse_vessel_mode(reader.value_or<std::string>("free_surface_vessel_model", "constantArea"));
    if (options.enabled && options.mode == FreeSurfaceMode::PlanarVolumeBudget &&
        options.vessel.mode == VesselVolumeMapMode::ConstantArea)
    {
        options.vessel.bottom_elevation = reader.required<real_t>("free_surface_bottom_elevation");
        options.vessel.top_elevation = reader.required<real_t>("free_surface_top_elevation");
        options.vessel.cross_section_area = reader.required<real_t>("free_surface_cross_section_area");
    }
    else
    {
        options.vessel.bottom_elevation =
            reader.value_or<real_t>("free_surface_bottom_elevation", options.vessel.bottom_elevation);
        options.vessel.top_elevation =
            reader.value_or<real_t>("free_surface_top_elevation", options.vessel.top_elevation);
        options.vessel.cross_section_area =
            reader.value_or<real_t>("free_surface_cross_section_area", options.vessel.cross_section_area);
    }
    options.vessel.height_table = reader.value_or<ArrReal>("free_surface_height_table", {});
    options.vessel.volume_table = reader.value_or<ArrReal>("free_surface_volume_table", {});
    options.vessel.total_internal_volume =
        reader.value_or<real_t>("free_surface_total_internal_volume", options.vessel.total_internal_volume);

    options.liquid_mass.mode = parse_liquid_volume_mode(
        reader.value_or<std::string>("free_surface_liquid_volume_model", "globalConstantMass"));
    options.liquid_mass.depletion_policy = options.range_policy;
    if (reader.contains("free_surface_initial_liquid_mass"))
    {
        options.liquid_mass.initial_liquid_mass = reader.required<real_t>("free_surface_initial_liquid_mass");
    }

    options.headspace.mode =
        parse_headspace_mode(reader.value_or<std::string>("free_surface_headspace_model", "vented"));
    options.headspace.temperature_mode =
        parse_temperature_mode(reader.value_or<std::string>("free_surface_headspace_temperature_model", "fixed"));
    options.headspace.ambient_pressure =
        reader.value_or<real_t>("free_surface_ambient_pressure", options.headspace.ambient_pressure);
    options.headspace.initial_pressure =
        reader.value_or<real_t>("free_surface_initial_pressure", options.headspace.initial_pressure);
    options.headspace.initial_temperature =
        reader.value_or<real_t>("free_surface_initial_temperature", options.headspace.initial_temperature);
    options.headspace.gas_constant =
        reader.value_or<real_t>("free_surface_headspace_gas_constant", options.headspace.gas_constant);
    options.headspace.compressibility_factor = reader.value_or<real_t>(
        "free_surface_headspace_compressibility_factor", options.headspace.compressibility_factor);
    options.headspace.prescribed_temperature_times =
        reader.value_or<ArrReal>("free_surface_headspace_temperature_times", {});
    options.headspace.prescribed_temperature_values =
        reader.value_or<ArrReal>("free_surface_headspace_temperature_values", {});
    options.headspace.total_internal_volume = options.vessel.total_internal_volume;

    const auto has_scalar_moles = reader.contains("free_surface_initial_headspace_moles");
    const auto has_species = reader.contains("free_surface_headspace_species");
    const auto has_species_moles = reader.contains("free_surface_initial_headspace_species_moles");
    if (has_scalar_moles && (has_species || has_species_moles))
    {
        throw std::invalid_argument("Configure either scalar or per-species initial headspace moles, not both.");
    }
    if (has_species != has_species_moles)
    {
        throw std::invalid_argument("free_surface_headspace_species and free_surface_initial_headspace_species_moles "
                                    "must be supplied together.");
    }
    if (has_scalar_moles)
    {
        options.headspace.initial_moles["gas"] = reader.required<real_t>("free_surface_initial_headspace_moles");
        options.headspace.infer_initial_moles = false;
    }
    else if (has_species)
    {
        const auto species = reader.required<ArrString>("free_surface_headspace_species");
        const auto moles = reader.required<ArrReal>("free_surface_initial_headspace_species_moles");
        if (species.size() != moles.size() || species.empty())
        {
            throw std::invalid_argument("Initial headspace species and mole arrays must have the same nonzero size.");
        }
        for (size_t index = 0; index < species.size(); ++index)
        {
            if (!options.headspace.initial_moles.emplace(species[index], moles[index]).second)
            {
                throw std::invalid_argument("Initial headspace species names must be unique.");
            }
        }
        options.headspace.infer_initial_moles = false;
    }

    options.coupling.maximum_correctors =
        reader.value_or<int>("free_surface_coupling_max_correctors", options.coupling.maximum_correctors);
    options.coupling.absolute_tolerance =
        reader.value_or<real_t>("free_surface_coupling_absolute_tolerance", options.coupling.absolute_tolerance);
    options.coupling.relative_tolerance =
        reader.value_or<real_t>("free_surface_coupling_relative_tolerance", options.coupling.relative_tolerance);
    options.coupling.relaxation =
        reader.value_or<real_t>("free_surface_coupling_relaxation", options.coupling.relaxation);
    options.coupling.minimum_absolute_pressure =
        reader.value_or<real_t>("free_surface_minimum_absolute_pressure", options.coupling.minimum_absolute_pressure);
    options.coupling.maximum_absolute_pressure =
        reader.value_or<real_t>("free_surface_maximum_absolute_pressure", options.coupling.maximum_absolute_pressure);
    options.coupling.volume_absolute_tolerance = reader.value_or<real_t>(
        "free_surface_volume_closure_absolute_tolerance", options.coupling.volume_absolute_tolerance);
    options.coupling.volume_relative_tolerance = reader.value_or<real_t>(
        "free_surface_volume_closure_relative_tolerance", options.coupling.volume_relative_tolerance);
    options.coupling.gas_absolute_tolerance =
        reader.value_or<real_t>("free_surface_gas_closure_absolute_tolerance", options.coupling.gas_absolute_tolerance);
    options.coupling.gas_relative_tolerance =
        reader.value_or<real_t>("free_surface_gas_closure_relative_tolerance", options.coupling.gas_relative_tolerance);

    validate_free_surface_options(options);
    return options;
}

void validate_free_surface_options(const FreeSurfaceOptions& options)
{
    if (options.mode == FreeSurfaceMode::PlanarALE)
    {
        throw std::invalid_argument(std::string(planar_ale_unavailable_diagnostic));
    }
    if (!std::isfinite(options.validity_warning_relative_level_change) ||
        options.validity_warning_relative_level_change < 0.0)
    {
        throw std::invalid_argument("Free-surface validity warning threshold must be finite and non-negative.");
    }
    validate_coupling_options(options.coupling);
    if (!options.enabled)
    {
        return;
    }
    if (options.mode != FreeSurfaceMode::PlanarVolumeBudget)
    {
        throw std::invalid_argument("free_surface_enabled=true requires free_surface_model='planarVolumeBudget'.");
    }
    if (options.liquid_mass.mode != LiquidVolumeMode::GlobalConstantMass &&
        options.liquid_mass.mode != LiquidVolumeMode::CellMassInventory)
    {
        throw std::invalid_argument("free_surface_liquid_volume_model is invalid.");
    }
    if (options.liquid_mass.mode == LiquidVolumeMode::CellMassInventory &&
        options.liquid_mass.depletion_policy != FreeSurfaceRangePolicy::Error)
    {
        throw std::invalid_argument("free_surface_liquid_volume_model 'cellMassInventory' currently requires "
                                    "free_surface_overflow_policy='error'; local phase acceptance cannot be "
                                    "conservatively clamped after boiling has accepted vapor.");
    }
    if (options.liquid_mass.initial_liquid_mass &&
        (!std::isfinite(*options.liquid_mass.initial_liquid_mass) || *options.liquid_mass.initial_liquid_mass < 0.0))
    {
        throw std::invalid_argument("free_surface_initial_liquid_mass must be finite and non-negative [kg].");
    }

    const auto volume_map = make_vessel_volume_map(options);
    if (options.initial_liquid_volume &&
        (!std::isfinite(*options.initial_liquid_volume) || *options.initial_liquid_volume < 0.0))
    {
        throw std::invalid_argument("free_surface_initial_liquid_volume must be finite and non-negative [m^3].");
    }
    if (options.initial_clear_level && !std::isfinite(*options.initial_clear_level))
    {
        throw std::invalid_argument("free_surface_initial_clear_level must be finite [m].");
    }
    if (!options.liquid_mass.initial_liquid_mass && !options.initial_liquid_volume && !options.initial_clear_level)
    {
        throw std::invalid_argument("Configure free_surface_initial_liquid_mass, free_surface_initial_liquid_volume, "
                                    "or free_surface_initial_clear_level; Milestone A does not assume a full vessel.");
    }
    if (options.initial_liquid_volume)
    {
        static_cast<void>(volume_map->evaluateLevelForVolume(*options.initial_liquid_volume));
    }
    if (options.initial_clear_level)
    {
        static_cast<void>(volume_map->evaluateVolumeBelow(*options.initial_clear_level));
    }
    if (options.initial_liquid_volume && options.initial_clear_level)
    {
        const auto volume_from_level = volume_map->volumeBelow(*options.initial_clear_level);
        const auto scale =
            std::max(1.0, std::max(std::abs(volume_from_level), std::abs(*options.initial_liquid_volume)));
        if (std::abs(volume_from_level - *options.initial_liquid_volume) > 1.0e-12 * scale)
        {
            throw std::invalid_argument(
                "Configured initial liquid volume and clear level are inconsistent with the vessel map.");
        }
    }
    const auto usable_volume = volume_map->totalUsableVolume();
    const auto total_internal_volume =
        std::isfinite(options.vessel.total_internal_volume) ? options.vessel.total_internal_volume : usable_volume;
    if (!finite_positive(total_internal_volume) || total_internal_volume < usable_volume)
    {
        throw std::invalid_argument("free_surface_total_internal_volume must be finite, positive, and no smaller than "
                                    "the mapped usable volume [m^3].");
    }

    auto headspace = options.headspace;
    if (std::isfinite(headspace.total_internal_volume))
    {
        const auto scale = std::max({1.0, std::abs(headspace.total_internal_volume), std::abs(total_internal_volume)});
        if (std::abs(headspace.total_internal_volume - total_internal_volume) >
            64.0 * std::numeric_limits<real_t>::epsilon() * scale)
        {
            throw std::invalid_argument("Headspace and vessel total internal volumes must agree when both are set.");
        }
    }
    headspace.total_internal_volume = total_internal_volume;
    validate_headspace_options(headspace);
}

std::shared_ptr<const VesselVolumeMap> make_vessel_volume_map(const FreeSurfaceOptions& options)
{
    if (options.vessel.mode == VesselVolumeMapMode::ConstantArea)
    {
        return std::make_shared<ConstantAreaVesselVolumeMap>(options.vessel.bottom_elevation,
            options.vessel.top_elevation, options.vessel.cross_section_area, options.range_policy);
    }
    return std::make_shared<TabulatedVesselVolumeMap>(
        options.vessel.height_table, options.vessel.volume_table, options.range_policy);
}

std::optional<real_t> configured_initial_liquid_volume(const FreeSurfaceOptions& options)
{
    if (options.initial_liquid_volume)
    {
        return options.initial_liquid_volume;
    }
    if (options.initial_clear_level)
    {
        return make_vessel_volume_map(options)->volumeBelow(*options.initial_clear_level);
    }
    return std::nullopt;
}

real_t prescribed_headspace_temperature(const HeadspaceOptions& options, real_t time)
{
    if (options.temperature_mode != HeadspaceTemperatureMode::Prescribed)
    {
        throw std::invalid_argument("prescribed_headspace_temperature requires the prescribed temperature model.");
    }
    validate_headspace_options(options);
    if (!std::isfinite(time))
    {
        throw std::invalid_argument("Headspace temperature interpolation time must be finite [s].");
    }
    const auto& times = options.prescribed_temperature_times;
    const auto& values = options.prescribed_temperature_values;
    if (time < times.front() || time > times.back())
    {
        throw std::out_of_range("Headspace temperature interpolation time is outside the prescribed history.");
    }
    if (time == times.front())
    {
        return values.front();
    }
    if (time == times.back())
    {
        return values.back();
    }
    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    const auto upper_index = static_cast<size_t>(std::distance(times.begin(), upper));
    const auto lower_index = upper_index - 1;
    const auto fraction = (time - times[lower_index]) / (times[upper_index] - times[lower_index]);
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index]);
}

HeadspaceModel::HeadspaceModel(HeadspaceOptions options) : d_options(std::move(options))
{
    validate_headspace_options(d_options);
    d_state.pressure =
        d_options.mode == HeadspaceMode::Vented ? d_options.ambient_pressure : d_options.initial_pressure;
    d_state.temperature = d_options.initial_temperature;
}

const HeadspaceOptions& HeadspaceModel::options() const noexcept
{
    return d_options;
}

const HeadspaceState& HeadspaceModel::state() const noexcept
{
    return d_state;
}

real_t HeadspaceModel::temperature(real_t supplied_temperature) const
{
    if (d_options.temperature_mode == HeadspaceTemperatureMode::Fixed)
    {
        return d_options.initial_temperature;
    }
    if (!finite_positive(supplied_temperature))
    {
        const auto source =
            d_options.temperature_mode == HeadspaceTemperatureMode::Prescribed ? "prescribed" : "bulk-liquid";
        throw std::invalid_argument(
            std::string("A finite positive ") + source + " headspace temperature [K] is required for this update.");
    }
    return supplied_temperature;
}

real_t HeadspaceModel::availableVolume(real_t pool_volume) const
{
    if (!std::isfinite(pool_volume) || pool_volume < 0.0)
    {
        throw std::invalid_argument("Pool volume must be finite and non-negative [m^3].");
    }
    return d_options.total_internal_volume - pool_volume;
}

void HeadspaceModel::setState(HeadspaceState state) noexcept
{
    d_state = state;
}

VentedHeadspaceModel::VentedHeadspaceModel(HeadspaceOptions options) : HeadspaceModel(std::move(options))
{
    if (d_options.mode != HeadspaceMode::Vented)
    {
        throw std::invalid_argument("VentedHeadspaceModel requires vented options.");
    }
}

std::unique_ptr<HeadspaceModel> VentedHeadspaceModel::clone() const
{
    return std::make_unique<VentedHeadspaceModel>(*this);
}

HeadspaceMode VentedHeadspaceModel::mode() const noexcept
{
    return HeadspaceMode::Vented;
}

void VentedHeadspaceModel::initialize(real_t pool_volume, real_t supplied_temperature)
{
    setState(trialState(pool_volume, supplied_temperature, {}));
}

HeadspaceState VentedHeadspaceModel::trialState(
    real_t pool_volume, real_t supplied_temperature, const GasMolesBySpecies& escaped_moles) const
{
    static_cast<void>(sum_moles(escaped_moles));
    return {.pressure = d_options.ambient_pressure,
        .volume = std::max(0.0, availableVolume(pool_volume)),
        .temperature = temperature(supplied_temperature),
        .total_moles = 0.0};
}

void VentedHeadspaceModel::commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles)
{
    if (state.pressure != d_options.ambient_pressure || !std::isfinite(state.volume) || state.volume < 0.0 ||
        !finite_positive(state.temperature))
    {
        throw std::invalid_argument("Invalid thermodynamic state supplied to vented headspace.");
    }
    auto prospective = d_vented_moles;
    add_moles(prospective, escaped_moles);
    d_vented_moles = std::move(prospective);
    setState(state);
}

const GasMolesBySpecies& VentedHeadspaceModel::headspaceMoles() const noexcept
{
    return d_empty_headspace;
}

const GasMolesBySpecies& VentedHeadspaceModel::ventedMoles() const noexcept
{
    return d_vented_moles;
}

ClosedIdealGasHeadspaceModel::ClosedIdealGasHeadspaceModel(HeadspaceOptions options)
    : HeadspaceModel(std::move(options))
{
    if (d_options.mode != HeadspaceMode::Closed)
    {
        throw std::invalid_argument("ClosedIdealGasHeadspaceModel requires closed options.");
    }
}

std::unique_ptr<HeadspaceModel> ClosedIdealGasHeadspaceModel::clone() const
{
    return std::make_unique<ClosedIdealGasHeadspaceModel>(*this);
}

HeadspaceMode ClosedIdealGasHeadspaceModel::mode() const noexcept
{
    return HeadspaceMode::Closed;
}

void ClosedIdealGasHeadspaceModel::initialize(real_t pool_volume, real_t supplied_temperature)
{
    const auto headspace_volume = availableVolume(pool_volume);
    const auto headspace_temperature = temperature(supplied_temperature);
    if (!(headspace_volume > 0.0))
    {
        throw std::domain_error("Closed headspace initialization requires positive headspace volume; the vessel is "
                                "overfilled or has zero headspace.");
    }
    GasMolesBySpecies initial_moles;
    if (d_options.infer_initial_moles)
    {
        initial_moles = {{"initialHeadspace",
            d_options.initial_pressure * headspace_volume /
                (d_options.compressibility_factor * d_options.gas_constant * headspace_temperature)}};
    }
    else
    {
        initial_moles = d_options.initial_moles;
    }
    const auto total_moles = sum_moles(initial_moles);
    const auto pressure = total_moles * d_options.compressibility_factor * d_options.gas_constant *
                          headspace_temperature / headspace_volume;
    if (!finite_positive(pressure))
    {
        throw std::overflow_error("Closed-headspace initial pressure is not finite and positive.");
    }
    const HeadspaceState initialized{.pressure = pressure,
        .volume = headspace_volume,
        .temperature = headspace_temperature,
        .total_moles = total_moles};
    if (!d_options.infer_initial_moles)
    {
        const auto scale = std::max({1.0, std::abs(initialized.pressure), std::abs(d_options.initial_pressure)});
        const auto tolerance = 1.0e-8 + 1.0e-10 * scale;
        if (std::abs(initialized.pressure - d_options.initial_pressure) > tolerance)
        {
            throw std::invalid_argument(
                "Explicit closed-headspace initial moles are inconsistent with the configured initial pressure, "
                "temperature, and available volume.");
        }
    }
    d_headspace_moles = std::move(initial_moles);
    setState(initialized);
}

HeadspaceState ClosedIdealGasHeadspaceModel::trialState(
    real_t pool_volume, real_t supplied_temperature, const GasMolesBySpecies& escaped_moles) const
{
    const auto volume = availableVolume(pool_volume);
    if (!(volume > 0.0))
    {
        throw std::domain_error(
            "Closed headspace requires positive headspace volume; the vessel is overfilled or has zero headspace.");
    }
    const auto gas_temperature = temperature(supplied_temperature);
    const auto prospective_moles = added_moles(d_headspace_moles, escaped_moles);
    const auto total_moles = sum_moles(prospective_moles);
    if (!(total_moles > 0.0))
    {
        throw std::domain_error(
            "Closed headspace requires a positive gas inventory to maintain positive absolute pressure.");
    }
    const auto pressure =
        total_moles * d_options.compressibility_factor * d_options.gas_constant * gas_temperature / volume;
    if (!finite_positive(pressure))
    {
        throw std::overflow_error("Closed-headspace ideal-gas pressure is not finite and positive.");
    }
    return {.pressure = pressure, .volume = volume, .temperature = gas_temperature, .total_moles = total_moles};
}

void ClosedIdealGasHeadspaceModel::commit(const HeadspaceState& state, const GasMolesBySpecies& escaped_moles)
{
    if (!finite_positive(state.pressure) || !finite_positive(state.volume) || !finite_positive(state.temperature) ||
        !finite_positive(state.total_moles))
    {
        throw std::invalid_argument("Invalid thermodynamic state supplied to closed headspace.");
    }
    auto prospective = d_headspace_moles;
    add_moles(prospective, escaped_moles);
    const auto expected_total = sum_moles(prospective);
    const auto expected_pressure =
        expected_total * d_options.compressibility_factor * d_options.gas_constant * state.temperature / state.volume;
    const auto tolerance = 1.0e-8 + 1.0e-10 * std::max(expected_pressure, state.pressure);
    if (std::abs(expected_total - state.total_moles) > 1.0e-12 * std::max(1.0, expected_total) ||
        std::abs(expected_pressure - state.pressure) > tolerance)
    {
        throw std::invalid_argument("Closed headspace state does not satisfy its ideal-gas inventory closure.");
    }
    d_headspace_moles = std::move(prospective);
    setState(state);
}

const GasMolesBySpecies& ClosedIdealGasHeadspaceModel::headspaceMoles() const noexcept
{
    return d_headspace_moles;
}

const GasMolesBySpecies& ClosedIdealGasHeadspaceModel::ventedMoles() const noexcept
{
    return d_empty_vent;
}

std::unique_ptr<HeadspaceModel> make_headspace_model(const FreeSurfaceOptions& options)
{
    auto headspace = options.headspace;
    // validate_free_surface_options() requires any explicit headspace value to
    // agree with the vessel value.  Keep the vessel as the single authoritative
    // capacity used at runtime as well as during validation.
    headspace.total_internal_volume = std::isfinite(options.vessel.total_internal_volume)
                                          ? options.vessel.total_internal_volume
                                          : configured_usable_volume(options.vessel);
    if (headspace.mode == HeadspaceMode::Vented)
    {
        return std::make_unique<VentedHeadspaceModel>(std::move(headspace));
    }
    return std::make_unique<ClosedIdealGasHeadspaceModel>(std::move(headspace));
}

PlanarFreeSurfaceModel::PlanarFreeSurfaceModel(std::shared_ptr<const VesselVolumeMap> volume_map,
    std::unique_ptr<HeadspaceModel> headspace, FreeSurfaceCouplingOptions coupling,
    real_t validity_warning_relative_level_change, real_t configured_level_underflow, real_t configured_level_overflow)
    : d_volume_map(std::move(volume_map)), d_headspace(std::move(headspace)), d_coupling(coupling),
      d_validity_warning_relative_level_change(validity_warning_relative_level_change),
      d_configured_level_underflow(configured_level_underflow), d_configured_level_overflow(configured_level_overflow)
{
    if (!d_volume_map || !d_headspace)
    {
        throw std::invalid_argument("PlanarFreeSurfaceModel requires a volume map and headspace model.");
    }
    validate_coupling_options(d_coupling);
    if (!std::isfinite(d_validity_warning_relative_level_change) || d_validity_warning_relative_level_change < 0.0)
    {
        throw std::invalid_argument("Free-surface validity warning threshold must be finite and non-negative.");
    }
    if (!std::isfinite(d_configured_level_underflow) || d_configured_level_underflow < 0.0 ||
        !std::isfinite(d_configured_level_overflow) || d_configured_level_overflow < 0.0)
    {
        throw std::invalid_argument("Configured initial level range diagnostics must be finite and non-negative [m].");
    }
}

void PlanarFreeSurfaceModel::validateUpdate(const FreeSurfaceUpdate& update, bool initializing) const
{
    if (!update.liquid_volume_at_pressure || !update.bubble_volume_at_pressure)
    {
        throw std::invalid_argument("Free-surface liquid- and bubble-volume callbacks are required.");
    }
    validate_gas_inventory(update.gas);
    if (!std::isfinite(update.time) || update.time < 0.0 || !std::isfinite(update.time_step) || update.time_step < 0.0)
    {
        throw std::invalid_argument("Free-surface accepted time and time step must be finite and non-negative [s].");
    }
    if (!std::isfinite(update.minimum_valid_absolute_pressure) || update.minimum_valid_absolute_pressure < 0.0)
    {
        throw std::invalid_argument(
            "Free-surface callback minimum valid absolute pressure must be finite and non-negative [Pa].");
    }
    if (!std::isfinite(update.liquid_volume_deficit) || update.liquid_volume_deficit < 0.0)
    {
        throw std::invalid_argument("Free-surface liquid-volume deficit must be finite and non-negative [m^3].");
    }
    if (initializing && sum_moles(update.gas.escaped_moles_this_step) != 0.0)
    {
        throw std::invalid_argument("Initial free-surface state cannot contain escaped-moles-this-step; initialize "
                                    "current compartments first.");
    }
    if (!initializing && !update.gas.initial_moles.empty() && update.gas.initial_moles != d_initial_gas_moles)
    {
        throw std::invalid_argument("Free-surface initial gas inventory is immutable after initialization.");
    }
}

void PlanarFreeSurfaceModel::initialize(const FreeSurfaceUpdate& update)
{
    if (d_initialized)
    {
        throw std::logic_error("PlanarFreeSurfaceModel is already initialized.");
    }
    validateUpdate(update, true);
    auto rollback_headspace = d_headspace->clone();
    try
    {
        const auto initial_pressure = d_headspace->state().pressure;
        const auto raw_liquid = checked_volume(update.liquid_volume_at_pressure, initial_pressure, "Liquid volume",
            d_volume_map->rangePolicy() == FreeSurfaceRangePolicy::ClampAndReport);
        const auto liquid_range = d_volume_map->evaluateLevelForVolume(raw_liquid);
        const auto liquid = raw_liquid < 0.0 ? liquid_range.accepted : raw_liquid;
        const auto bubble =
            checked_volume(update.bubble_volume_at_pressure, initial_pressure, "Submerged bubble volume", false);
        d_headspace->initialize(liquid + bubble, update.headspace_temperature);

        const auto closure = solveClosure(update);
        d_headspace->commit(closure.headspace, {});

        if (!update.gas.initial_moles.empty())
        {
            d_initial_gas_moles = update.gas.initial_moles;
        }
        else
        {
            d_initial_gas_moles = update.gas.dissolved_moles;
            add_moles(d_initial_gas_moles, update.gas.submerged_moles);
            add_moles(d_initial_gas_moles, d_headspace->headspaceMoles());
            add_moles(d_initial_gas_moles, d_headspace->ventedMoles());
        }
        d_diagnostics = makeDiagnostics(closure, update, true);
        d_initial_pool_level = d_diagnostics.pool_level;
        d_diagnostics.old_pool_level = d_diagnostics.pool_level;
        d_diagnostics.old_clear_level = d_diagnostics.clear_level;
        d_initialized = true;
    }
    catch (...)
    {
        d_headspace = std::move(rollback_headspace);
        d_initial_gas_moles.clear();
        d_committed_escaped_moles.clear();
        d_diagnostics = {};
        d_initial_pool_level = 0.0;
        d_initialized = false;
        throw;
    }
}

void PlanarFreeSurfaceModel::update(const FreeSurfaceUpdate& update)
{
    if (!d_initialized)
    {
        throw std::logic_error("PlanarFreeSurfaceModel must be initialized before update.");
    }
    validateUpdate(update, false);
    const auto closure = solveClosure(update);
    auto diagnostics = makeDiagnostics(closure, update, false);
    d_headspace->commit(closure.headspace, update.gas.escaped_moles_this_step);
    add_moles(d_committed_escaped_moles, update.gas.escaped_moles_this_step);
    d_diagnostics = std::move(diagnostics);
}

PlanarFreeSurfaceModel::ClosureResult PlanarFreeSurfaceModel::solveClosure(const FreeSurfaceUpdate& update) const
{
    return d_headspace->mode() == HeadspaceMode::Vented ? solveVented(update) : solveClosed(update);
}

PlanarFreeSurfaceModel::ClosureResult PlanarFreeSurfaceModel::solveVented(const FreeSurfaceUpdate& update) const
{
    ClosureResult result;
    const auto pressure = d_headspace->options().ambient_pressure;
    if (pressure < update.minimum_valid_absolute_pressure)
    {
        throw std::runtime_error(
            "Vented free-surface pressure is below the valid domain of a coupled volume callback.");
    }
    result.liquid_volume = checked_volume(update.liquid_volume_at_pressure, pressure, "Liquid volume",
        d_volume_map->rangePolicy() == FreeSurfaceRangePolicy::ClampAndReport);
    const auto liquid_evaluation = d_volume_map->evaluateLevelForVolume(result.liquid_volume);
    if (result.liquid_volume < 0.0)
    {
        result.liquid_volume = liquid_evaluation.accepted;
        result.dryout_deficit = liquid_evaluation.underflow;
    }
    result.bubble_volume = checked_volume(update.bubble_volume_at_pressure, pressure, "Submerged bubble volume", false);
    result.headspace = d_headspace->trialState(
        result.liquid_volume + result.bubble_volume, update.headspace_temperature, update.gas.escaped_moles_this_step);
    result.iterations = 1;
    return result;
}

PlanarFreeSurfaceModel::ClosureResult PlanarFreeSurfaceModel::evaluateClosedTrial(
    const FreeSurfaceUpdate& update, real_t pressure) const
{
    ClosureResult result;
    result.liquid_volume = checked_volume(update.liquid_volume_at_pressure, pressure, "Liquid volume",
        d_volume_map->rangePolicy() == FreeSurfaceRangePolicy::ClampAndReport);
    if (result.liquid_volume < 0.0)
    {
        result.dryout_deficit = -result.liquid_volume;
        result.liquid_volume = 0.0;
    }
    result.bubble_volume = checked_volume(update.bubble_volume_at_pressure, pressure, "Submerged bubble volume", false);
    const auto pool_volume = result.liquid_volume + result.bubble_volume;
    const auto available = d_headspace->options().total_internal_volume - pool_volume;
    if (!(available > 0.0))
    {
        result.headspace = {.pressure = pressure,
            .volume = available,
            .temperature = d_headspace->state().temperature,
            .total_moles = 0.0};
        result.residual = -std::numeric_limits<real_t>::infinity();
        return result;
    }
    const auto equilibrium =
        d_headspace->trialState(pool_volume, update.headspace_temperature, update.gas.escaped_moles_this_step);
    result.residual = pressure - equilibrium.pressure;
    result.headspace = equilibrium;
    result.headspace.pressure = pressure;
    return result;
}

PlanarFreeSurfaceModel::ClosureResult PlanarFreeSurfaceModel::solveClosed(const FreeSurfaceUpdate& update) const
{
    auto lower_pressure = std::max(d_coupling.minimum_absolute_pressure, update.minimum_valid_absolute_pressure);
    auto upper_pressure = d_coupling.maximum_absolute_pressure;
    if (!(lower_pressure < upper_pressure))
    {
        throw std::runtime_error(
            "Closed free-surface closure has no pressure interval shared by its configured bounds and callbacks.");
    }
    auto lower = evaluateClosedTrial(update, lower_pressure);
    auto upper = evaluateClosedTrial(update, upper_pressure);

    auto converged_at = [this](ClosureResult result, real_t pressure)
    {
        const auto scale = std::max(1.0, std::max(std::abs(pressure), std::abs(pressure - result.residual)));
        const auto configured_tolerance = d_coupling.absolute_tolerance + d_coupling.relative_tolerance * scale;
        const auto thermodynamic_tolerance = 0.5 * (1.0e-8 + 1.0e-10 * scale);
        const auto tolerance = std::min(configured_tolerance, thermodynamic_tolerance);
        if (std::isfinite(result.residual) && std::abs(result.residual) <= tolerance)
        {
            return std::optional<ClosureResult>{std::move(result)};
        }
        return std::optional<ClosureResult>{};
    };
    if (auto result = converged_at(lower, lower_pressure))
    {
        return *result;
    }
    if (auto result = converged_at(upper, upper_pressure))
    {
        return *result;
    }

    if (lower.residual > 0.0)
    {
        throw std::runtime_error("Closed free-surface closure root is below the configured minimum absolute pressure.");
    }
    if (!(upper.residual >= 0.0))
    {
        throw std::runtime_error("Closed free-surface closure root is above the configured maximum absolute pressure, "
                                 "or no positive headspace volume exists at that bound.");
    }

    ClosureResult best = std::abs(upper.residual) < std::abs(lower.residual) ? upper : lower;
    for (int iteration = 1; iteration <= d_coupling.maximum_correctors; ++iteration)
    {
        real_t proposed = 0.5 * (lower_pressure + upper_pressure);
        if (std::isfinite(lower.residual) && std::isfinite(upper.residual) && upper.residual != lower.residual)
        {
            const auto secant =
                (lower_pressure * upper.residual - upper_pressure * lower.residual) / (upper.residual - lower.residual);
            const auto margin = 0.1 * (upper_pressure - lower_pressure);
            if (secant > lower_pressure + margin && secant < upper_pressure - margin)
            {
                proposed = secant;
            }
        }
        const auto midpoint = 0.5 * (lower_pressure + upper_pressure);
        const auto pressure = d_coupling.relaxation * proposed + (1.0 - d_coupling.relaxation) * midpoint;
        auto current = evaluateClosedTrial(update, pressure);
        current.iterations = iteration;
        const auto scale = std::max(1.0, std::max(std::abs(pressure), std::abs(pressure - current.residual)));
        const auto configured_tolerance = d_coupling.absolute_tolerance + d_coupling.relative_tolerance * scale;
        const auto thermodynamic_tolerance = 0.5 * (1.0e-8 + 1.0e-10 * scale);
        const auto tolerance = std::min(configured_tolerance, thermodynamic_tolerance);
        if (std::isfinite(current.residual) && std::abs(current.residual) <= tolerance)
        {
            return current;
        }
        if (std::abs(current.residual) < std::abs(best.residual))
        {
            best = current;
        }
        if (current.residual < 0.0)
        {
            lower_pressure = pressure;
            lower = current;
        }
        else
        {
            upper_pressure = pressure;
            upper = current;
        }
    }

    std::ostringstream message;
    message << "Closed free-surface closure failed to converge after " << d_coupling.maximum_correctors
            << " correctors; pressure bracket [" << lower_pressure << ", " << upper_pressure << "] Pa, best residual "
            << best.residual << " Pa.";
    throw std::runtime_error(message.str());
}

FreeSurfaceDiagnostics PlanarFreeSurfaceModel::makeDiagnostics(
    const ClosureResult& closure, const FreeSurfaceUpdate& update, bool initializing) const
{
    const auto& gas = update.gas;
    FreeSurfaceDiagnostics diagnostics;
    diagnostics.time = update.time;
    diagnostics.time_step = update.time_step;
    diagnostics.liquid_volume = closure.liquid_volume;
    diagnostics.submerged_bubble_volume = closure.bubble_volume;
    diagnostics.pool_volume = closure.liquid_volume + closure.bubble_volume;

    const auto clear = d_volume_map->evaluateLevelForVolume(diagnostics.liquid_volume);
    const auto pool = d_volume_map->evaluateLevelForVolume(diagnostics.pool_volume);
    diagnostics.clear_level = clear.value;
    diagnostics.pool_level = pool.value;
    diagnostics.dryout_deficit = std::max(closure.dryout_deficit, clear.underflow);
    diagnostics.dryout_deficit = std::max(diagnostics.dryout_deficit, update.liquid_volume_deficit);
    diagnostics.overflow_volume = std::max(clear.overflow, pool.overflow);
    diagnostics.configured_level_underflow = d_configured_level_underflow;
    diagnostics.configured_level_overflow = d_configured_level_overflow;
    if (d_headspace->mode() == HeadspaceMode::Closed && diagnostics.overflow_volume > 0.0)
    {
        throw std::domain_error("Closed free-surface closure cannot accept vessel-map overfill; positive headspace and "
                                "an in-range pool level are required.");
    }
    diagnostics.surface_area = d_volume_map->areaAt(diagnostics.pool_level);
    diagnostics.volume_closure_residual = d_volume_map->volumeBelow(diagnostics.pool_level) - diagnostics.pool_volume;
    const auto volume_scale =
        std::max(std::abs(diagnostics.pool_volume), std::abs(d_volume_map->volumeBelow(diagnostics.pool_level)));
    diagnostics.normalized_volume_closure_residual = diagnostics.volume_closure_residual / std::max(1.0, volume_scale);
    if (diagnostics.overflow_volume == 0.0 && diagnostics.dryout_deficit == 0.0)
    {
        const auto volume_tolerance =
            d_coupling.volume_absolute_tolerance + d_coupling.volume_relative_tolerance * volume_scale;
        if (std::abs(diagnostics.volume_closure_residual) > volume_tolerance)
        {
            throw std::runtime_error("Planar free-surface volume closure exceeded its configured tolerance.");
        }
    }
    diagnostics.headspace = closure.headspace;
    diagnostics.nonlinear_iterations = closure.iterations;
    diagnostics.nonlinear_residual = closure.residual;

    if (initializing)
    {
        diagnostics.old_clear_level = diagnostics.clear_level;
        diagnostics.old_pool_level = diagnostics.pool_level;
    }
    else
    {
        diagnostics.old_clear_level = d_diagnostics.clear_level;
        diagnostics.old_pool_level = d_diagnostics.pool_level;
        if (update.time_step > 0.0)
        {
            diagnostics.clear_level_rate = (diagnostics.clear_level - diagnostics.old_clear_level) / update.time_step;
            diagnostics.pool_level_rate = (diagnostics.pool_level - diagnostics.old_pool_level) / update.time_step;
        }
        const auto vessel_height = d_volume_map->topElevation() - d_volume_map->bottomElevation();
        diagnostics.relative_level_change = std::abs(diagnostics.pool_level - d_initial_pool_level) / vessel_height;
        diagnostics.validity_warning = diagnostics.relative_level_change > d_validity_warning_relative_level_change;
    }

    diagnostics.generated_gas_moles = gas.generated_moles;
    diagnostics.dissolved_gas_moles = gas.dissolved_moles;
    diagnostics.submerged_gas_moles = gas.submerged_moles;
    diagnostics.submerged_population_gas_moles = gas.submerged_population_moles;
    diagnostics.escaped_gas_moles_this_step = gas.escaped_moles_this_step;
    diagnostics.other_sink_gas_moles = gas.other_sink_moles;
    diagnostics.headspace_gas_moles = d_headspace->headspaceMoles();
    diagnostics.vented_gas_moles = d_headspace->ventedMoles();
    if (d_headspace->mode() == HeadspaceMode::Closed)
    {
        add_moles(diagnostics.headspace_gas_moles, gas.escaped_moles_this_step);
    }
    else
    {
        add_moles(diagnostics.vented_gas_moles, gas.escaped_moles_this_step);
    }
    diagnostics.gas_closure_by_species =
        gas_closure(d_initial_gas_moles, gas, diagnostics.headspace_gas_moles, diagnostics.vented_gas_moles);
    auto species_value = [](const GasMolesBySpecies& values, const std::string& species)
    {
        const auto iterator = values.find(species);
        return iterator == values.end() ? 0.0 : iterator->second;
    };
    if (!gas.submerged_population_moles.empty())
    {
        GasMolesBySpecies population_totals;
        for (const auto& [population, moles] : gas.submerged_population_moles)
        {
            static_cast<void>(population);
            add_moles(population_totals, moles);
        }
        std::set<std::string> population_species;
        for (const auto& [species, value] : population_totals)
        {
            static_cast<void>(value);
            population_species.insert(species);
        }
        for (const auto& [species, value] : gas.submerged_moles)
        {
            static_cast<void>(value);
            population_species.insert(species);
        }
        for (const auto& species : population_species)
        {
            const auto aggregate = species_value(gas.submerged_moles, species);
            const auto populations = species_value(population_totals, species);
            const auto scale = std::max({1.0, std::abs(aggregate), std::abs(populations)});
            const auto tolerance = d_coupling.gas_absolute_tolerance + d_coupling.gas_relative_tolerance * scale;
            if (std::abs(aggregate - populations) > tolerance)
            {
                throw std::runtime_error("Free-surface submerged population moles do not sum to the species total.");
            }
        }
    }
    for (const auto& [species, residual] : diagnostics.gas_closure_by_species)
    {
        const auto inventory_scale =
            species_value(d_initial_gas_moles, species) + species_value(gas.generated_moles, species) +
            species_value(gas.dissolved_moles, species) + species_value(gas.submerged_moles, species) +
            species_value(diagnostics.headspace_gas_moles, species) +
            species_value(diagnostics.vented_gas_moles, species) + species_value(gas.other_sink_moles, species);
        const auto gas_tolerance =
            d_coupling.gas_absolute_tolerance + d_coupling.gas_relative_tolerance * inventory_scale;
        diagnostics.normalized_gas_closure_by_species[species] = residual / std::max(1.0, inventory_scale);
        if (std::abs(residual) > gas_tolerance)
        {
            throw std::runtime_error(
                "Free-surface gas-inventory closure for species '" + species + "' exceeded its configured tolerance.");
        }
    }
    diagnostics.gas_closure_residual =
        std::accumulate(diagnostics.gas_closure_by_species.begin(), diagnostics.gas_closure_by_species.end(), 0.0,
            [](real_t sum, const auto& entry) { return sum + entry.second; });
    diagnostics.normalized_gas_closure_residual = std::accumulate(diagnostics.normalized_gas_closure_by_species.begin(),
        diagnostics.normalized_gas_closure_by_species.end(), 0.0,
        [](real_t maximum, const auto& entry) { return std::max(maximum, std::abs(entry.second)); });
    return diagnostics;
}

bool PlanarFreeSurfaceModel::initialized() const noexcept
{
    return d_initialized;
}

FreeSurfaceDiagnostics PlanarFreeSurfaceModel::diagnostics() const
{
    if (!d_initialized)
    {
        throw std::logic_error("PlanarFreeSurfaceModel diagnostics require initialization.");
    }
    return d_diagnostics;
}

real_t PlanarFreeSurfaceModel::headspacePressure() const noexcept
{
    return d_headspace->state().pressure;
}

const VesselVolumeMap& PlanarFreeSurfaceModel::volumeMap() const noexcept
{
    return *d_volume_map;
}

const HeadspaceModel& PlanarFreeSurfaceModel::headspace() const noexcept
{
    return *d_headspace;
}

const GasMolesBySpecies& PlanarFreeSurfaceModel::committedEscapedMoles() const noexcept
{
    return d_committed_escaped_moles;
}

std::unique_ptr<PlanarFreeSurfaceModel> make_planar_free_surface_model(const FreeSurfaceOptions& options)
{
    validate_free_surface_options(options);
    if (!options.enabled)
    {
        return nullptr;
    }
    auto volume_map = make_vessel_volume_map(options);
    real_t configured_level_underflow = 0.0;
    real_t configured_level_overflow = 0.0;
    if (options.initial_clear_level)
    {
        const auto range = volume_map->evaluateVolumeBelow(*options.initial_clear_level);
        configured_level_underflow = range.underflow;
        configured_level_overflow = range.overflow;
    }
    return std::make_unique<PlanarFreeSurfaceModel>(std::move(volume_map), make_headspace_model(options),
        options.coupling, options.validity_warning_relative_level_change, configured_level_underflow,
        configured_level_overflow);
}

} // namespace SimpleFluid
