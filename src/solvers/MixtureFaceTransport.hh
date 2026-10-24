/** @file MixtureFaceTransport.hh
 * @brief Supplied same-carrier mixture diffusion and explicit Fourier face-contact laws.
 */
#pragma once
#include "dataclass/TpetraTypes.hh"
#include "equations/SpeciesDiffusion.hh"
#include <functional>
#include <optional>
#include <vector>

namespace SimpleFluid::Reacting
{
using MixtureFaceGlobalId = DefaultTpetraTypes::global_ordinal_type;
/** Actual post-chemistry physical phase evaluated at the current thermal guess.
 * phase_volume is m/rho(T,p,x); advected_material_volume is never a denominator.
 * An absent phase has zero volume/concentrations; its EOS and h laws are not called.
 */
struct MixtureFacePhaseSample
{
    double temperature = 0, absolute_pressure = 0, phase_volume = 0, density = 0;
    double bulk_volume = 0; ///< Current geometric CV for explicit contact laws, never the concentration denominator.
    std::vector<double> physical_moles, mole_fractions, concentrations;
    double counter_moles = 0;
};
struct MixtureFaceContext
{
    /** Integer identities; never encoded as floating values. */
    MixtureFaceGlobalId algebraic_identity = 0, geometry_identity = 0;
    double area = 0, center_distance = 0, accepted_time = 0, time_step = 0;
    std::array<double, 3> center{}, normal{};
    const MixtureFacePhaseSample& owner;
    const MixtureFacePhaseSample& neighbor;
};
struct MixtureDiffusionLaw
{
    SpeciesDiffusionBinding binding; ///< counterSpecies must be explicitly selected, including an ordinary ID zero.
    std::vector<SpeciesId> primary_species;
    /** Catalog-ordered K_i [m3/s], e.g. supplied D_i*A_contact/d.
     * Explicit contact is part of this law. No phase connectivity is inferred.
     */
    std::function<std::vector<double>(const MixtureFaceContext&)> molar_conductance;
    /** Supplied partial-molar h [J/mol] for a donor at its actual T,p,x.
     * No bulk-mixture/ideal-gas surrogate is supplied by this adapter.
     */
    std::function<double(SpeciesId, const DiffusionCompartment&, const MixtureFacePhaseSample&)> partial_molar_enthalpy;
};
struct MixtureFourierLaw
{
    std::string identity;
    std::uint64_t revision = 1, schema_fingerprint = 0;
    /** Explicit contact G [W/K], e.g. supplied k*A_contact/d. Q=dt*G*(T_owner-T_neighbor).
     * Selecting a dispersed physical phase does not imply contact with any other bin.
     */
    std::function<double(const MixtureFaceContext&)> conductance;
};
struct MixtureFaceTransportOptions
{
    std::string identity;
    std::uint64_t revision = 1, schema_fingerprint = 0;
    std::size_t physical_phase = 0;
    std::optional<MixtureDiffusionLaw> diffusion;
    std::optional<MixtureFourierLaw> fourier;
};
} // namespace SimpleFluid::Reacting
