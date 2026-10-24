/** @file LowMachFaceTransportAdapter.cc
 * Conservative distributed source packets with explicit immutable supplied laws.
 */
#include "solvers/details/LowMachFaceTransportAdapter.hh"
#include "solvers/details/LowMachRawSpeciesInventory.hh"
#include "equations/CollectiveValidation.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_MultiVector.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <variant>

namespace SimpleFluid::Reacting::low_mach_faces
{
namespace
{
using Sum = SimpleFluid::detail::CompensatedSum<long double>;
using MV = Pack::multi_vector_type;
void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
double narrow(long double value)
{
    const double result = static_cast<double>(value);
    require(std::isfinite(value) && std::isfinite(result) && (value == 0 || result != 0),
        "Face transport amount/energy is not finitely representable without underflow.");
    return result;
}
bool same(const DiffusionCompartment& a, const DiffusionCompartment& b)
{
    return a.phase == b.phase && a.population == b.population;
}
StageStamp stamp(const Flow::SourceStageContext& context)
{
    const auto& s = context.transport;
    return {&context.mesh, s.accepted_revision, s.accepted_generation, s.old_geometry_epoch, s.new_geometry_epoch,
        s.schema_fingerprint, s.accepted_time, s.time_step, s.stage};
}
void uniform_text(const Mesh& mesh, const std::string& text)
{
    int size = 0;
    collective_detail::collective_local_validation(mesh, "Face law metadata size",
        [&]
        {
            require(text.size() <= std::size_t(std::numeric_limits<int>::max()), "Face law metadata is too large.");
            size = static_cast<int>(text.size());
        });
    collective_detail::require_uniform_value(mesh, size, "Face law metadata length");
    std::vector<int> local, minimum, maximum;
    collective_detail::collective_local_validation(mesh, "Face law metadata buffers",
        [&]
        {
            local.assign(text.begin(), text.end());
            minimum.resize(size);
            maximum.resize(size);
        });
    if (size > 0)
    {
        const auto comm = mesh.owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, size, local.data(), minimum.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, size, local.data(), maximum.data());
    }
    require(minimum == maximum, "Face law identities/schema/bindings must agree on every rank.");
}
std::uint64_t integer_identity(GO id)
{
    if constexpr (std::numeric_limits<GO>::is_signed)
        require(id >= 0, "Negative canonical face identity.");
    return static_cast<std::uint64_t>(id);
}
/** Read compact native descriptors only: never materialize the handle's
 * compatibility indexer or enumerate global cells/faces. Composite region
 * layouts and periodic interface descriptors are already retained metadata.
 */
template<class Geometry> bool has_native_periodicity(const Geometry& geometry)
{
    if constexpr (requires { geometry.indexer().periodic_dimensions[0]; })
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (geometry.indexer().periodic_dimensions[axis])
                return true;
    if constexpr (requires { geometry.indexer().axial_periodic; })
        if (geometry.indexer().axial_periodic)
            return true;
    if constexpr (requires { geometry.layout().periodic[0]; })
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (geometry.layout().periodic[axis])
                return true;
    if constexpr (requires {
                      geometry.interfaces();
                      geometry.regions();
                  })
    {
        for (const auto& connection : geometry.interfaces())
            if (std::visit(
                    [](const auto& interface)
                    {
                        if constexpr (requires { interface.periodic_translation; })
                            return bool(interface.periodic_translation);
                        else
                            return false;
                    },
                    connection))
                return true;
        for (const auto& region : geometry.regions())
            if (std::visit([](const auto& provider) { return has_native_periodicity(provider.topology()); }, region))
                return true;
    }
    return false;
}
void require_nonperiodic_native_topology(const Mesh& mesh)
{
    require(!mesh.visit([](const auto& geometry) { return has_native_periodicity(geometry); }),
        "Mixture face transport requires nonperiodic topology; native periodic seams are unsupported.");
}
CellSpeciesState zero_like(const CellSpeciesState& state)
{
    auto result = state;
    std::ranges::fill(result.amounts, 0.);
    for (auto& p : result.dispersedPopulations)
        std::ranges::fill(p.amounts, 0.);
    return result;
}
} // namespace

LowMachFaceTransportAdapter::LowMachFaceTransportAdapter(std::shared_ptr<const Mesh> mesh,
    std::shared_ptr<const SpeciesCatalog> catalog, Options options, PhaseBindings compartments)
    : mesh_(std::move(mesh)), catalog_(std::move(catalog)), options_(std::move(options)),
      compartments_(std::move(compartments))
{
    if (!mesh_)
        throw std::invalid_argument("Face adapter requires a communicator-bearing mesh.");
    std::string signature;
    collective_detail::collective_local_validation(*mesh_, "Face adapter immutable configuration",
        [&]
        {
            require(catalog_ && !options_.identity.empty() && options_.revision > 0 &&
                        options_.schema_fingerprint == catalog_->fingerprint() &&
                        (options_.diffusion || options_.fourier),
                "Face adapter needs immutable identity/revision/schema and explicit laws.");
            const auto ns = catalog_->species().size();
            require(ns <= (std::numeric_limits<std::size_t>::max() - 8) / 3 &&
                        options_.physical_phase < compartments_.size(),
                "Face packet schema/phase is invalid.");
            std::set<std::tuple<SpeciesId, SpeciesPhase, std::string>> roles;
            for (const auto& phase : compartments_)
            {
                require(phase.size() == ns, "Face phase bindings must explicitly cover catalog order.");
                for (std::size_t s = 0; s < ns; ++s)
                    require(static_cast<std::size_t>(phase[s].phase) < 4 &&
                                (phase[s].population.empty() || phase[s].phase == SpeciesPhase::Dispersed) &&
                                roles.emplace(catalog_->species()[s].id, phase[s].phase, phase[s].population).second,
                        "Face physical phases must bind unique valid chemical compartments.");
            }
            std::ostringstream text;
            text << options_.identity.size() << ':' << options_.identity << ':' << options_.revision << ':'
                 << options_.schema_fingerprint << ':' << options_.physical_phase << ':' << compartments_.size();
            for (const auto& phase : compartments_)
                for (const auto& c : phase)
                    text << ':' << int(c.phase) << ':' << c.population.size() << ':' << c.population;
            text << ':' << bool(options_.diffusion) << ':' << bool(options_.fourier);
            if (options_.diffusion)
            {
                const auto& d = *options_.diffusion;
                const auto& b = d.binding;
                require(!b.identity.empty() && b.revision > 0 && b.schemaFingerprint == catalog_->fingerprint() &&
                            d.molar_conductance && d.partial_molar_enthalpy && !d.primary_species.empty(),
                    "Diffusion needs explicit coefficients, donor partial h and immutable metadata.");
                require(b.energyConvention == EnergyConvention::SensibleEnergy ||
                            b.energyConvention == EnergyConvention::IncludesFormationEnergy,
                    "Invalid diffusion caloric convention.");
                require(b.counterSpecies.has_value(), "Diffusion requires explicit finite counter-species selection.");
                const auto counter = catalog_->index(*b.counterSpecies);
                require(same(compartments_[options_.physical_phase][counter], b.counter) &&
                            catalog_->allows(*b.counterSpecies, b.counter.phase),
                    "Finite counter must belong to the same carrier.");
                std::set<SpeciesId> primary;
                for (const auto id : d.primary_species)
                {
                    const auto s = catalog_->index(id);
                    require(primary.insert(id).second &&
                                same(compartments_[options_.physical_phase][s], b.transported) &&
                                catalog_->allows(id, b.transported.phase) &&
                                !(same(b.transported, b.counter) && id == *b.counterSpecies),
                        "Primary diffusion species/binding is invalid or duplicates its counter.");
                }
                text << ':' << b.identity.size() << ':' << b.identity << ':' << b.revision << ':' << b.schemaFingerprint
                     << ':' << int(b.energyConvention) << ':' << int(b.transported.phase) << ':'
                     << b.transported.population.size() << ':' << b.transported.population << ':'
                     << int(b.counter.phase) << ':' << b.counter.population.size() << ':' << b.counter.population << ':'
                     << *b.counterSpecies;
                text << ':' << d.primary_species.size();
                for (const auto id : d.primary_species)
                    text << ':' << id;
            }
            if (options_.fourier)
            {
                const auto& h = *options_.fourier;
                require(!h.identity.empty() && h.revision > 0 && h.schema_fingerprint == catalog_->fingerprint() &&
                            h.conductance,
                    "Fourier contact requires explicit immutable conductance metadata.");
                text << ':' << h.identity.size() << ':' << h.identity << ':' << h.revision << ':'
                     << h.schema_fingerprint;
            }
            require_nonperiodic_native_topology(*mesh_);
            signature = text.str();
            owned_cells_ = mesh_->owned_cell_map();
            overlap_cells_ = mesh_->overlap_cell_map();
            owned_faces_ = mesh_->owned_face_map();
            overlap_faces_ = mesh_->overlap_face_map();
            require(!owned_cells_.is_null() && !overlap_cells_.is_null() && !owned_faces_.is_null() &&
                        !overlap_faces_.is_null(),
                "Face adapter requires native field maps.");
        });
    uniform_text(*mesh_, signature);
    // Import construction and import calls are collective library seams, never inside a pure-local fence.
    cell_import_ = std::make_unique<Pack::import_type>(owned_cells_, overlap_cells_);
    face_import_ = std::make_unique<Pack::import_type>(owned_faces_, overlap_faces_);
}

Receipt LowMachFaceTransportAdapter::preview(
    const Flow::SourceStageContext& context, std::span<const CellSpeciesState> post_chemistry) const
{
    return preview(context, post_chemistry, context.thermodynamic_trial.cells);
}

Receipt LowMachFaceTransportAdapter::preview(const Flow::SourceStageContext& context,
    std::span<const CellSpeciesState> post_chemistry, std::span<const Flow::CellState> prepared_caloric_guesses) const
{
    return preview_impl(context, post_chemistry, prepared_caloric_guesses, {}, false);
}
Receipt LowMachFaceTransportAdapter::preview(const Flow::SourceStageContext& context,
    std::span<const CellSpeciesState> post_chemistry, std::span<const Flow::CellState> prepared_caloric_guesses,
    std::span<const std::vector<std::array<long double, 2>>> owner_parts) const
{
    return preview_impl(context, post_chemistry, prepared_caloric_guesses, owner_parts, true);
}
Receipt LowMachFaceTransportAdapter::preview_impl(const Flow::SourceStageContext& context,
    std::span<const CellSpeciesState> post_chemistry, std::span<const Flow::CellState> prepared_caloric_guesses,
    std::span<const std::vector<std::array<long double, 2>>> owner_parts, bool raw_authority) const
{
    collective_detail::require_uniform_value(*mesh_, int(raw_authority), "Face raw predecessor authority");
    const auto ns = catalog_->species().size(), np = compartments_.size(), physical = options_.physical_phase;
    const std::size_t sample_columns = 2 * ns + 6, packet_columns = 3 * ns + 4;
    const std::size_t t = 2 * ns, p = t + 1, volume = t + 2, density = t + 3;
    const std::size_t counter_amount = t + 4, bulk_volume = t + 5;
    const std::size_t h_owner = 1 + ns, h_neighbor = 1 + 2 * ns;
    const std::size_t hc_owner = 1 + 3 * ns, hc_neighbor = hc_owner + 1, heat_column = hc_owner + 2;
    std::unique_ptr<MV> owned_samples, local_samples, owned_packets, local_packets;
    Receipt result;
    collective_detail::collective_local_validation(*mesh_, "Face stage state and halo packing",
        [&]
        {
            require(
                &context.mesh == mesh_.get() && context.transport.new_geometry_epoch == mesh_->geometry_epoch() &&
                    context.transport.schema_fingerprint == catalog_->fingerprint() &&
                    context.transport.time_step > 0 && std::isfinite(context.transport.time_step) &&
                    std::isfinite(context.transport.accepted_time) &&
                    std::isfinite(context.transport.accepted_time + context.transport.time_step) &&
                    context.transport.accepted_time + context.transport.time_step > context.transport.accepted_time &&
                    !context.transport.stage.provider_identity.empty() &&
                    context.transport.stage.provider_revision > 0 &&
                    context.transport.new_control_volumes.size() == mesh_->num_owned_cells() &&
                    context.thermodynamic_trial.cells.size() == mesh_->num_owned_cells() &&
                    prepared_caloric_guesses.size() == mesh_->num_owned_cells() &&
                    post_chemistry.size() == mesh_->num_owned_cells(),
                "Face stage is stale, foreign or malformed.");
            if (raw_authority)
            {
                require(owner_parts.size() == post_chemistry.size(), "Raw face predecessor owned rows differ.");
                for (std::size_t c = 0; c < post_chemistry.size(); ++c)
                    validate_raw_inventory(*catalog_, post_chemistry[c], owner_parts[c]);
            }
            require(mesh_->owned_cell_map().getRawPtr() == owned_cells_.getRawPtr() &&
                        mesh_->overlap_cell_map().getRawPtr() == overlap_cells_.getRawPtr() &&
                        mesh_->owned_face_map().getRawPtr() == owned_faces_.getRawPtr() &&
                        mesh_->overlap_face_map().getRawPtr() == overlap_faces_.getRawPtr(),
                "Face stage field maps changed.");
            require_nonperiodic_native_topology(*mesh_);
            result.producer_ = this;
            result.stamp = stamp(context);
            result.physical_phase = physical;
            owned_samples = std::make_unique<MV>(owned_cells_, sample_columns, true);
            local_samples = std::make_unique<MV>(overlap_cells_, sample_columns, true);
            owned_packets = std::make_unique<MV>(owned_faces_, packet_columns, true);
            local_packets = std::make_unique<MV>(overlap_faces_, packet_columns, true);
            for (std::size_t c = 0; c < post_chemistry.size(); ++c)
            {
                post_chemistry[c].validate(*catalog_);
                const auto bound = [&](std::size_t s, const DiffusionCompartment& role)
                {
                    return std::ranges::any_of(
                        compartments_, [&](const auto& physical_roles) { return same(physical_roles[s], role); });
                };
                for (std::size_t s = 0; s < ns; ++s)
                {
                    const auto id = catalog_->species()[s].id;
                    for (std::size_t phase = 0; phase < 4; ++phase)
                    {
                        const DiffusionCompartment role{static_cast<SpeciesPhase>(phase), {}};
                        require(post_chemistry[c].amount(*catalog_, id, role.phase) == 0 || bound(s, role),
                            "A positive chemical inventory is absent from the physical phase binding.");
                    }
                    for (const auto& population : post_chemistry[c].dispersedPopulations)
                    {
                        const DiffusionCompartment role{SpeciesPhase::Dispersed, population.id};
                        require(population.amounts[s] == 0 || bound(s, role),
                            "A positive dispersed inventory is absent from the physical phase binding.");
                    }
                }
                const auto cell = static_cast<LO>(c);
                require(context.transport.new_control_volumes[c] == mesh_->cell_volume(cell),
                    "Face stage uses current trial volumes, never accepted-old or advected phase volumes.");
                const auto& phases = prepared_caloric_guesses[c].phases;
                const auto& original_phases = context.thermodynamic_trial.cells[c].phases;
                require(phases.size() == np && original_phases.size() == np && phases[physical].thermodynamics &&
                            phases[physical].thermodynamics.get() == original_phases[physical].thermodynamics.get() &&
                            phases[physical].thermodynamics->catalog().fingerprint() == catalog_->fingerprint(),
                    "Face stage physical EOS/schema binding is invalid.");
                if (options_.diffusion)
                    require(phases[physical].thermodynamics->energy_convention() ==
                                options_.diffusion->binding.energyConvention,
                        "Diffusion h and phase energy conventions differ.");
                const auto row = owned_cells_->getLocalElement(mesh_->cell_global_id(cell));
                require(row != Teuchos::OrdinalTraits<LO>::invalid(), "Face stage owned cell map is inconsistent.");
                std::vector<double> amounts(ns), x(ns);
                Sum total, mass;
                for (std::size_t s = 0; s < ns; ++s)
                {
                    const auto& role = compartments_[physical][s];
                    amounts[s] =
                        post_chemistry[c].amount(*catalog_, catalog_->species()[s].id, role.phase, role.population);
                    total += amounts[s];
                    mass += static_cast<long double>(amounts[s]) * catalog_->species()[s].molarMass;
                    owned_samples->replaceLocalValue(row, s, amounts[s]);
                }
                double phase_volume = 0, phase_density = 0, temperature = 0;
                const double pressure = phases[physical].absolute_pressure;
                require(std::isfinite(pressure), "Face candidate phase pressure is nonfinite.");
                if (total.value() > 0)
                {
                    temperature = phases[physical].temperature;
                    require(std::isfinite(temperature) && temperature > 0 && pressure > 0,
                        "Occupied post-chemistry phase requires a supplied valid thermal guess.");
                    for (std::size_t s = 0; s < ns; ++s)
                        x[s] = narrow(amounts[s] / total.value());
                    const auto point = phases[physical].thermodynamics->evaluate(temperature, pressure, x);
                    phase_density = point.density;
                    phase_volume = narrow(mass.value() / phase_density);
                    require(phase_volume > 0, "Post-chemistry EOS phase volume is invalid.");
                }
                for (std::size_t s = 0; s < ns; ++s)
                {
                    double concentration = 0;
                    if (options_.diffusion && phase_volume > 0)
                    {
                        const auto& role = options_.diffusion->binding.transported;
                        // Only selected species need this compartment; other species can forbid it.
                        if (std::ranges::find(options_.diffusion->primary_species, catalog_->species()[s].id) !=
                            options_.diffusion->primary_species.end())
                            concentration = narrow(static_cast<long double>(post_chemistry[c].amount(*catalog_,
                                                       catalog_->species()[s].id, role.phase, role.population)) /
                                                   phase_volume);
                    }
                    owned_samples->replaceLocalValue(row, ns + s, concentration);
                }
                owned_samples->replaceLocalValue(row, t, temperature);
                owned_samples->replaceLocalValue(row, p, pressure);
                owned_samples->replaceLocalValue(row, volume, phase_volume);
                owned_samples->replaceLocalValue(row, density, phase_density);
                double counter = 0;
                if (options_.diffusion)
                {
                    const auto& b = options_.diffusion->binding;
                    counter =
                        post_chemistry[c].amount(*catalog_, *b.counterSpecies, b.counter.phase, b.counter.population);
                }
                owned_samples->replaceLocalValue(row, counter_amount, counter);
                owned_samples->replaceLocalValue(row, bulk_volume, context.transport.new_control_volumes[c]);
            }
        });
    local_samples->doImport(*owned_samples, *cell_import_, Tpetra::REPLACE);
    collective_detail::collective_local_validation(*mesh_, "Face canonical owner law evaluation",
        [&]
        {
            const auto sample = [&](LO cell)
            {
                const auto row = overlap_cells_->getLocalElement(mesh_->cell_global_id(cell));
                require(
                    row != Teuchos::OrdinalTraits<LO>::invalid(), "Interior face requires its one-layer cell halo.");
                SideSample side;
                side.physical_moles.resize(ns);
                side.mole_fractions.resize(ns);
                side.concentrations.resize(ns);
                Sum amount;
                for (std::size_t s = 0; s < ns; ++s)
                {
                    side.physical_moles[s] = local_samples->getData(s)[row];
                    side.concentrations[s] = local_samples->getData(ns + s)[row];
                    amount += side.physical_moles[s];
                }
                if (amount.value() > 0)
                    for (std::size_t s = 0; s < ns; ++s)
                        side.mole_fractions[s] = narrow(side.physical_moles[s] / amount.value());
                side.temperature = local_samples->getData(t)[row];
                side.absolute_pressure = local_samples->getData(p)[row];
                side.phase_volume = local_samples->getData(volume)[row];
                side.density = local_samples->getData(density)[row];
                side.counter_moles = local_samples->getData(counter_amount)[row];
                side.bulk_volume = local_samples->getData(bulk_volume)[row];
                return side;
            };
            for (std::size_t f = 0; f < mesh_->num_faces(); ++f)
            {
                const auto face = static_cast<LO>(f);
                if (!mesh_->is_owned_face(face))
                    continue;
                const auto owner = mesh_->owner_cell(face), neighbor = mesh_->neighbor_cell(face);
                require(owner != Mesh::invalid_local_id(), "Canonical face owner cell is absent.");
                if (mesh_->is_geometry_exterior_face(face))
                {
                    require(mesh_->opposite_or_periodic_neighbor_cell(face, owner) == Mesh::invalid_local_id(),
                        "This no-flux-boundary draft rejects periodic face pairs.");
                    continue;
                }
                require(neighbor != Mesh::invalid_local_id() && mesh_->is_owned_cell(owner),
                    "Canonical event owner must own the source-oriented owner cell.");
                const auto a = sample(owner), b = sample(neighbor);
                const auto center = mesh_->face_centroid(face), normal = mesh_->face_normal(face);
                const FaceContext law{.algebraic_identity = mesh_->face_global_id(face),
                    .geometry_identity = mesh_->face_geometry_global_id(face),
                    .area = mesh_->face_area(face),
                    .center_distance = mesh_->face_cell_center_distance(face),
                    .accepted_time = context.transport.accepted_time,
                    .time_step = context.transport.time_step,
                    .center = {center.x, center.y, center.z},
                    .normal = {normal.x, normal.y, normal.z},
                    .owner = a,
                    .neighbor = b};
                require(std::ranges::all_of(law.center, [](double x) { return std::isfinite(x); }) &&
                            std::ranges::all_of(law.normal, [](double x) { return std::isfinite(x); }) &&
                            std::isfinite(law.area) && law.area > 0 && std::isfinite(law.center_distance) &&
                            law.center_distance > 0,
                    "Face law geometry is invalid.");
                const auto row = owned_faces_->getLocalElement(law.algebraic_identity);
                require(row != Teuchos::OrdinalTraits<LO>::invalid(), "Canonical face owned map is inconsistent.");
                owned_packets->replaceLocalValue(row, 0, 1.); // Numerical active flag only, never an encoded entity ID.
                if (options_.diffusion)
                {
                    const auto& d = *options_.diffusion;
                    const auto conductance = d.molar_conductance(law);
                    require(conductance.size() == ns, "Molar conductance must use catalog order.");
                    std::vector<double> flux(ns);
                    Sum primary_mass;
                    for (std::size_t s = 0; s < ns; ++s)
                    {
                        const bool selected =
                            std::ranges::find(d.primary_species, catalog_->species()[s].id) != d.primary_species.end();
                        require(
                            std::isfinite(conductance[s]) && conductance[s] >= 0 && (selected || conductance[s] == 0),
                            "Molar conductance is invalid or selects an undeclared species.");
                        require(conductance[s] == 0 || (a.phase_volume > 0 && b.phase_volume > 0),
                            "Finite same-carrier diffusion contact requires both physical carrier phases.");
                        flux[s] = narrow(static_cast<long double>(law.time_step) * conductance[s] *
                                         (static_cast<long double>(a.concentrations[s]) - b.concentrations[s]));
                        owned_packets->replaceLocalValue(row, 1 + s, flux[s]);
                        primary_mass += static_cast<long double>(flux[s]) * catalog_->species()[s].molarMass;
                        if (flux[s] != 0)
                        {
                            const double h = d.partial_molar_enthalpy(
                                catalog_->species()[s].id, d.binding.transported, flux[s] > 0 ? a : b);
                            require(std::isfinite(h), "Supplied donor partial-molar enthalpy is nonfinite.");
                            owned_packets->replaceLocalValue(row, (flux[s] > 0 ? h_owner : h_neighbor) + s, h);
                        }
                    }
                    const auto counter = catalog_->index(*d.binding.counterSpecies);
                    const double n = primary_mass.value() == 0
                                         ? 0
                                         : narrow(-primary_mass.value() / catalog_->species()[counter].molarMass);
                    if (n != 0)
                    {
                        const double h =
                            d.partial_molar_enthalpy(*d.binding.counterSpecies, d.binding.counter, n > 0 ? a : b);
                        require(std::isfinite(h), "Supplied finite-counter partial-molar enthalpy is nonfinite.");
                        owned_packets->replaceLocalValue(row, n > 0 ? hc_owner : hc_neighbor, h);
                    }
                }
                if (options_.fourier)
                {
                    const double g = options_.fourier->conductance(law);
                    require(std::isfinite(g) && g >= 0 && (g == 0 || (a.phase_volume > 0 && b.phase_volume > 0)),
                        "Fourier contact requires finite nonnegative conductance and two occupied selected phases.");
                    const double q = narrow(static_cast<long double>(law.time_step) * g *
                                            (static_cast<long double>(a.temperature) - b.temperature));
                    owned_packets->replaceLocalValue(row, heat_column, q);
                }
            }
        });
    local_packets->doImport(*owned_packets, *face_import_, Tpetra::REPLACE);
    collective_detail::collective_local_validation(*mesh_, "Face owned donor reservation and source assembly",
        [&]
        {
            for (std::size_t f = 0; f < mesh_->num_faces(); ++f)
            {
                const auto face = static_cast<LO>(f);
                if (mesh_->is_geometry_exterior_face(face))
                    continue;
                const auto owner = mesh_->owner_cell(face), neighbor = mesh_->neighbor_cell(face);
                const bool own_a = owner != Mesh::invalid_local_id() && mesh_->is_owned_cell(owner);
                const bool own_b = neighbor != Mesh::invalid_local_id() && mesh_->is_owned_cell(neighbor);
                if (!own_a && !own_b)
                    continue; // Outer halo geometry is not an owned physical event.
                require(owner != Mesh::invalid_local_id() && neighbor != Mesh::invalid_local_id(),
                    "Owned incident face lacks its one-layer donor halo.");
                const auto row = overlap_faces_->getLocalElement(mesh_->face_global_id(face));
                require(row != Teuchos::OrdinalTraits<LO>::invalid() && local_packets->getData(0)[row] == 1.,
                    "Canonical face packet was not imported from its event owner.");
                const auto a = own_a ? std::optional<std::size_t>(owner) : std::nullopt;
                const auto b = own_b ? std::optional<std::size_t>(neighbor) : std::nullopt;
                if (options_.diffusion)
                {
                    CanonicalDiffusionFace packet;
                    packet.identity = integer_identity(mesh_->face_geometry_global_id(face));
                    packet.owner = a;
                    packet.neighbor = b;
                    packet.moles.resize(ns);
                    packet.ownerMolarEnthalpy.resize(ns);
                    packet.neighborMolarEnthalpy.resize(ns);
                    for (std::size_t s = 0; s < ns; ++s)
                    {
                        packet.moles[s] = local_packets->getData(1 + s)[row];
                        packet.ownerMolarEnthalpy[s] = local_packets->getData(h_owner + s)[row];
                        packet.neighborMolarEnthalpy[s] = local_packets->getData(h_neighbor + s)[row];
                    }
                    packet.ownerCounterMolarEnthalpy = local_packets->getData(hc_owner)[row];
                    packet.neighborCounterMolarEnthalpy = local_packets->getData(hc_neighbor)[row];
                    // Both velocities remain identically zero: declared barycentric zero-mass convention.
                    result.raw_diffusion_faces.push_back(std::move(packet));
                }
                if (options_.fourier)
                    result.fourier_faces.push_back(
                        {mesh_->face_geometry_global_id(face), a, b, local_packets->getData(heat_column)[row]});
            }
            std::ranges::sort(result.raw_diffusion_faces, {}, &CanonicalDiffusionFace::identity);
            if (options_.diffusion)
                result.diffusion = raw_authority
                    ? preview_species_diffusion(*catalog_, post_chemistry, options_.diffusion->binding,
                          result.raw_diffusion_faces, owner_parts)
                    : preview_species_diffusion(
                          *catalog_, post_chemistry, options_.diffusion->binding, result.raw_diffusion_faces);
            else
            {
                result.diffusion.state.assign(post_chemistry.begin(), post_chemistry.end());
                result.diffusion.cells.resize(post_chemistry.size());
                for (std::size_t c = 0; c < post_chemistry.size(); ++c)
                    result.diffusion.cells[c].change = zero_like(post_chemistry[c]);
            }
            result.cells.resize(post_chemistry.size());
            result.rounded.sources.resize(post_chemistry.size());
            result.rounded.gross_events.assign(post_chemistry.size(), std::vector<double>(2 * ns + 6));
            std::vector<std::vector<Sum>> gross(post_chemistry.size(), std::vector<Sum>(2 * ns + 6));
            const auto add_material = [&](std::size_t c, SpeciesId id, const DiffusionCompartment& compartment,
                                          double value, std::size_t offset)
            {
                result.cells[c].material.push_back({id, compartment, value});
                gross[c][offset + (value < 0 ? 1 : 0)] += std::abs(static_cast<long double>(value));
            };
            for (std::size_t i = 0; i < result.raw_diffusion_faces.size(); ++i)
            {
                const auto& raw = result.raw_diffusion_faces[i];
                require(i < result.diffusion.faces.size() && result.diffusion.faces[i].identity == raw.identity,
                    "Canonical diffusion receipt is absent.");
                const auto& event = result.diffusion.faces[i];
                const auto& binding = options_.diffusion->binding;
                const auto deposit = [&](std::optional<std::size_t> c, double sign)
                {
                    if (!c)
                        return;
                    for (std::size_t s = 0; s < ns; ++s)
                        if (raw.moles[s] != 0)
                        {
                            add_material(
                                *c, catalog_->species()[s].id, binding.transported, sign * raw.moles[s], 2 * s);
                            result.cells[*c].diffusion_energy.push_back({sign * raw.moles[s],
                                raw.moles[s] > 0 ? raw.ownerMolarEnthalpy[s] : raw.neighborMolarEnthalpy[s]});
                        }
                    if (event.counterMoles != 0)
                    {
                        add_material(*c, *binding.counterSpecies, binding.counter, sign * event.counterMoles, 2 * ns);
                        result.cells[*c].diffusion_energy.push_back({sign * event.counterMoles,
                            event.counterMoles > 0 ? raw.ownerCounterMolarEnthalpy : raw.neighborCounterMolarEnthalpy});
                    }
                    const double q = sign * event.enthalpy;
                    gross[*c][2 * ns + 2 + (q < 0 ? 1 : 0)] += std::abs(static_cast<long double>(q));
                };
                deposit(event.owner, -1);
                deposit(event.neighbor, 1);
            }
            std::ranges::sort(result.fourier_faces, {}, &FourierFaceLedger::geometry_identity);
            for (const auto& face : result.fourier_faces)
            {
                const auto deposit = [&](std::optional<std::size_t> c, double sign)
                {
                    if (!c)
                        return;
                    const double q = sign * face.energy;
                    result.cells[*c].fourier_energy.push_back(q);
                    gross[*c][2 * ns + 4 + (q < 0 ? 1 : 0)] += std::abs(static_cast<long double>(q));
                };
                deposit(face.owner, -1);
                deposit(face.neighbor, 1);
            }
            for (std::size_t c = 0; c < post_chemistry.size(); ++c)
            {
                auto& source = result.rounded.sources[c];
                source.species_moles.assign(np, std::vector<double>(ns));
                source.internal_energy.assign(np, 0.);
                std::vector<Sum> material(ns);
                Sum energy;
                for (const auto& term : result.cells[c].material)
                    material[catalog_->index(term.species)] += term.moles;
                for (const auto& term : result.cells[c].diffusion_energy)
                    energy += static_cast<long double>(term.moles) * term.molar_enthalpy;
                for (const auto q : result.cells[c].fourier_energy)
                    energy += q;
                for (std::size_t s = 0; s < ns; ++s)
                    source.species_moles[physical][s] = narrow(material[s].value());
                source.internal_energy[physical] = narrow(energy.value());
                // source.momentum remains exactly zero; no rounding-based donor velocity cancellation.
                for (std::size_t i = 0; i < gross[c].size(); ++i)
                    result.rounded.gross_events[c][i] = narrow(gross[c][i].value());
            }
        });
    return result;
}

void LowMachFaceTransportAdapter::validate_receipt(
    const Receipt& receipt, const Flow::SourceStageContext& context) const
{
    collective_detail::collective_local_validation(*mesh_, "Face stage receipt replay identity",
        [&]
        {
            require(receipt.producer_ == this && receipt.physical_phase == options_.physical_phase &&
                        receipt.stamp == stamp(context) && &context.mesh == mesh_.get() &&
                        context.transport.new_geometry_epoch == mesh_->geometry_epoch(),
                "Face receipt belongs to another producer, physical phase, stage, geometry or provider invocation.");
        });
}
} // namespace SimpleFluid::Reacting::low_mach_faces
