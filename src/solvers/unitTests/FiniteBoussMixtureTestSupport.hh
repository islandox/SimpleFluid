/** @file FiniteBoussMixtureTestSupport.hh
 * @brief Manufactured solvent+A+B+G1+G2 reference-CV/cp*T fixture, not a material calibration.
 */
#pragma once
#include "ReactingMixtureTestSupport.hh"
#include "equations/BoussFiniteVapor.hh"

namespace SimpleFluid::Testing::FiniteBouss
{
namespace M = ReactingMixture;
namespace R = Reacting;
using P = R::SpeciesPhase;
struct Mixture
{
    std::shared_ptr<const R::SpeciesCatalog> catalog = M::make_catalog();
    std::shared_ptr<const R::ReactionMechanism> mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
        std::vector<R::ReactionDefinition>{{M::conversion, "finite_A_to_B_and_two_gases",
            {{M::reactant, P::Liquid, -1}, {M::product, P::Liquid, 1}, {M::gas_one, P::Resolved, 1},
                {M::gas_two, P::Resolved, 1}},
            0}},
        R::EnergyConvention::SensibleEnergy);
    std::shared_ptr<const R::BoussReferenceVolume> reference;
    Mixture()
    {
        auto value = std::make_shared<R::BoussReferenceVolume>();
        value->identity = "manufactured.finite_Bouss_reference";
        value->revision = 1;
        value->schemaFingerprint = catalog->fingerprint();
        value->liquidDensity = value->thermalDensity = 2;
        value->heatCapacity = 5;
        value->gasMolarVolumes.resize(catalog->species().size());
        value->gasMolarVolumes[catalog->index(M::solvent)] = .018; // W/rho_v with rho_v=1.
        value->gasMolarVolumes[catalog->index(M::gas_one)] = .1;
        value->gasMolarVolumes[catalog->index(M::gas_two)] = .2;
        reference = std::move(value);
    }
    ReactingGasOptions options(double alpha = 0, double maximum = .95) const
    {
        ReactingGasOptions result;
        result.mode = ReactingGasMode::Species;
        result.reaction_mechanism = mechanism;
        result.bouss_reference = reference;
        auto sources = std::make_shared<R::BoussCoordinatedSources>();
        sources->identity = "manufactured.coordinated_finite_sources";
        sources->revision = 1;
        sources->schemaFingerprint = catalog->fingerprint();
        result.bouss_sources = sources;
        result.alpha_max = maximum;
        result.reference_pressure = 100000;
        result.population_ids = {"micro", "large"};
        result.min_radius = 1e-10;
        result.max_radius = 10;
        result.initial_species_state = [schema = catalog, alpha](
                                           const R::ReactionEnvironment& environment, std::int64_t)
        {
            R::CellSpeciesState state(*schema);
            for (const auto* name : {"micro", "large"})
                state.dispersedPopulations.push_back({name, std::vector<double>(schema->species().size())});
            state.set_amount(*schema, M::solvent, P::Liquid, 2 * environment.bulkVolume / .018);
            state.set_amount(*schema, M::reactant, P::Liquid, environment.bulkVolume);
            state.set_amount(*schema, M::gas_one, P::Resolved, alpha * environment.bulkVolume / .1);
            return state;
        };
        result.initial_population_numbers = [](const R::ReactionContext&) { return std::vector<double>{0, 0}; };
        return result;
    }
    std::shared_ptr<R::BoussCoordinatedSources> vapor(const ReactingGasOptions& options, double latent = 1000) const
    {
        auto result = std::make_shared<R::BoussCoordinatedSources>(*options.bouss_sources);
        ++result->revision;
        result->vapor = R::BoussVaporBinding{M::solvent, P::Resolved, {}, 373, latent, 1, 0};
        return result;
    }
    std::shared_ptr<R::BoussCoordinatedSources> gas(const ReactingGasOptions& options, double rate) const
    {
        auto result = std::make_shared<R::BoussCoordinatedSources>(*options.bouss_sources);
        ++result->revision;
        result->gas = R::BoussGasAdmissionLaw{"manufactured.independent_molar_request", 1, catalog->fingerprint(),
            [rate](const R::ReactionContext& context, double dt)
            {
                return std::vector<R::IntegratedReactionExtent>{
                    {M::conversion, rate * context.environment.bulkVolume * dt}};
            }};
        return result;
    }
};
} // namespace SimpleFluid::Testing::FiniteBouss
