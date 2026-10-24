/** @file LowMachFaceSourceComposition.hh
 * root composition seam, intentionally independent of controller ownership.
 */
#pragma once
#include "equations/CollectiveValidation.hh"
#include "solvers/details/LowMachFaceTransportAdapter.hh"
#include "utils/CompensatedSum.hh"
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Reacting::low_mach_faces
{
/** One set of accumulators spans pure chemistry AND later face stages.
 * Do not narrow one stage and re-seed these sums from its rounded result.
 * Construct/reset and every mutation below are MPI-free and require the caller's local failure fence.
 */
class LocalSourceAccumulator
{
    using Sum = SimpleFluid::detail::CompensatedSum<long double>;
    std::vector<std::vector<std::vector<Sum>>> material_;
    std::vector<std::vector<Sum>> energy_;
    std::vector<Sum> declared_energy_;
    std::vector<std::array<Sum, 3>> momentum_;
    static double narrow(long double value)
    {
        const double result = static_cast<double>(value);
        if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0))
            throw std::invalid_argument("Composed source packet is nonfinite or unrepresentable.");
        return result;
    }

public:
    void reset(std::size_t cells, std::size_t phases, std::size_t species)
    {
        material_.assign(cells, std::vector<std::vector<Sum>>(phases, std::vector<Sum>(species)));
        energy_.assign(cells, std::vector<Sum>(phases));
        declared_energy_.assign(cells, {});
        momentum_.assign(cells, {});
    }
    void add_material(std::size_t c, std::size_t phase, std::size_t species, long double moles)
    {
        if (!std::isfinite(moles))
            throw std::invalid_argument("Nonfinite raw material source.");
        material_.at(c).at(phase).at(species) += moles;
    }
    /** Declare a physical raw energy channel before optional user allocation. */
    void declare_energy(std::size_t c, long double joules)
    {
        if (!std::isfinite(joules))
            throw std::invalid_argument("Nonfinite declared raw energy source.");
        declared_energy_.at(c) += joules;
    }
    /** Add only an already allocated user result; original channels must be declared separately. */
    void add_allocated_energy(std::size_t c, std::size_t phase, long double joules)
    {
        if (!std::isfinite(joules))
            throw std::invalid_argument("Nonfinite allocated energy source.");
        energy_.at(c).at(phase) += joules;
    }
    /** Default raw allocation both assigns and declares exactly the same physical channel. */
    void add_energy(std::size_t c, std::size_t phase, long double joules)
    {
        add_allocated_energy(c, phase, joules);
        declare_energy(c, joules);
    }
    void add_momentum(std::size_t c, std::size_t axis, long double impulse)
    {
        if (!std::isfinite(impulse))
            throw std::invalid_argument("Nonfinite raw external impulse.");
        momentum_.at(c).at(axis) += impulse;
    }
    void add_faces(const Receipt& receipt, const SpeciesCatalog& catalog)
    {
        if (receipt.cells.size() != material_.size())
            throw std::invalid_argument("Face receipt does not match source accumulator owned rows.");
        for (std::size_t c = 0; c < receipt.cells.size(); ++c)
        {
            for (const auto& term : receipt.cells[c].material)
                add_material(c, receipt.physical_phase, catalog.index(term.species), term.moles);
            for (const auto& term : receipt.cells[c].diffusion_energy)
                add_energy(c, receipt.physical_phase, static_cast<long double>(term.moles) * term.molar_enthalpy);
            for (const auto q : receipt.cells[c].fourier_energy)
                add_energy(c, receipt.physical_phase, q);
            // The same-carrier barycentric diffusion convention contributes exactly no impulse.
        }
    }
    std::vector<Flow::CellSources> finalize() const
    {
        std::vector<Flow::CellSources> result(material_.size());
        for (std::size_t c = 0; c < result.size(); ++c)
        {
            result[c].species_moles.resize(material_[c].size());
            result[c].internal_energy.resize(energy_[c].size());
            Sum applied_energy, applied_scale;
            for (std::size_t p = 0; p < material_[c].size(); ++p)
            {
                result[c].species_moles[p].resize(material_[c][p].size());
                for (std::size_t s = 0; s < material_[c][p].size(); ++s)
                    result[c].species_moles[p][s] = narrow(material_[c][p][s].value());
                result[c].internal_energy[p] = narrow(energy_[c][p].value());
                applied_energy += result[c].internal_energy[p];
                applied_scale += std::abs(static_cast<long double>(result[c].internal_energy[p]));
            }
            const long double declared = declared_energy_[c].value();
            Sum discrepancy;
            discrepancy += applied_energy.value();
            discrepancy += -declared;
            const auto bound =
                64 * std::numeric_limits<double>::epsilon() * (applied_scale.value() + std::abs(declared));
            if (!std::isfinite(declared) || !std::isfinite(applied_energy.value()) ||
                !std::isfinite(applied_scale.value()) || !std::isfinite(bound) || !std::isfinite(discrepancy.value()) ||
                std::abs(discrepancy.value()) > bound)
                throw std::invalid_argument("Final phase packets lost declared raw thermal energy.");
            for (std::size_t axis = 0; axis < 3; ++axis)
                result[c].momentum.component(axis) = narrow(momentum_[c][axis].value());
        }
        return result;
    }
};

/** Can one rounded signed packet represent the actual sequential endpoint?
 * The bound uses endpoint operands only: a large accepted inventory/gross turnover may not hide a lost small stock.
 * MPI-free; the controller invokes this inside its local failure fence before any later communication.
 */
inline void validate_source_endpoint(double transported, double packet, double actual)
{
    using Sum = SimpleFluid::detail::CompensatedSum<long double>;
    if (!std::isfinite(transported) || transported < 0 || !std::isfinite(packet) || !std::isfinite(actual) ||
        actual < 0)
        throw std::invalid_argument("Nonfinite or negative source endpoint input.");
    Sum replay;
    replay += transported;
    replay += packet;
    const auto recovered = replay.value();
    Sum error;
    error += recovered;
    error += -static_cast<long double>(actual);
    const auto bound = 64 * std::numeric_limits<double>::epsilon() *
                       (std::abs(recovered) + std::abs(static_cast<long double>(actual)));
    if (!std::isfinite(recovered) || recovered < 0 || (recovered == 0) != (actual == 0) ||
        !std::isfinite(error.value()) || !std::isfinite(bound) || std::abs(error.value()) > bound)
        throw std::invalid_argument("Rounded source packet cannot replay its actual sequential endpoint.");
}

struct LocalChemistryStage
{
    std::vector<CellSpeciesState> post_chemistry; ///< Copy the actual advance_reactions(...).state.
    std::vector<SpeciesSourceLedger> ledgers;
    std::vector<Flow::CellState> caloric_guesses; ///< Prepared candidate T/p/EOS; donor moles are separate.
    LocalSourceAccumulator sums; ///< Raw chemistry/external/allocated heat channels go here before any narrowing.
    std::vector<std::vector<double>> gross_events;
};
struct ComposedSourceStage
{
    Flow::SourceStageResult result;
    std::vector<SpeciesSourceLedger> chemistry_ledgers;
    std::vector<std::shared_ptr<const Receipt>> faces;
    std::vector<CellSpeciesState> post_faces; ///< Actual donors for the next interphase stage.
};

/** Exact apply order: transported inventory -> pure chemistry -> ordered face stages ->
 * later interphase stages -> one final source packet -> flow caloric/pressure/occupancy acceptance.
 * Each face stage receives the actual prior stage state; independent reservations
 * against one shared chemistry base are forbidden.
 *
 * prepare_chemistry is entirely MPI-free: copy actual advance_reactions(...).state,
 * retain ReactionAdvance::localReceipt channels (not rounded extent reports) in sums,
 * and provide prepared caloric T/p/EOS guesses.
 * No outer collective_local_validation may wrap this function: face adapters communicate.
 */
inline ComposedSourceStage compose_post_chemistry_faces(const Flow::SourceStageContext& context,
    const SpeciesCatalog& catalog, std::span<const LowMachFaceTransportAdapter* const> adapters,
    const std::function<void(const Flow::SourceStageContext&, LocalChemistryStage&)>& prepare_chemistry)
{
    collective_detail::require_uniform_value(
        context.mesh, static_cast<unsigned long long>(adapters.size()), "Ordered mixture face-stage count");
    LocalChemistryStage chemistry;
    ComposedSourceStage result;
    collective_detail::collective_local_validation(context.mesh, "Pure posttransport chemistry stage",
        [&]
        {
            for (const auto* adapter : adapters)
                if (!adapter)
                    throw std::invalid_argument("Null ordered face-stage adapter.");
            prepare_chemistry(context, chemistry);
            if (chemistry.post_chemistry.size() != context.mesh.num_owned_cells() ||
                chemistry.gross_events.size() != context.mesh.num_owned_cells() ||
                chemistry.caloric_guesses.size() != context.mesh.num_owned_cells())
                throw std::invalid_argument("Pure chemistry receipt must follow actual owned rows.");
            result.faces.reserve(adapters.size());
        });
    auto donors = std::move(chemistry.post_chemistry);
    for (const auto* adapter : adapters)
    {
        auto face = adapter->preview(context, donors, chemistry.caloric_guesses); // Collective internal stage.
        collective_detail::collective_local_validation(context.mesh, "Accumulate one ordered canonical face stage",
            [&]
            {
                chemistry.sums.add_faces(face, catalog);
                donors =
                    face.diffusion.state; // Actual finite state, never endpoint-differenced or original-base reuse.
                if (face.rounded.gross_events.size() != chemistry.gross_events.size())
                    throw std::invalid_argument("Face gross events do not match chemistry owned rows.");
                for (std::size_t c = 0; c < chemistry.gross_events.size(); ++c)
                {
                    const auto& extra = face.rounded.gross_events[c];
                    chemistry.gross_events[c].insert(chemistry.gross_events[c].end(), extra.begin(), extra.end());
                }
                result.faces.push_back(std::make_shared<const Receipt>(std::move(face)));
            });
    }
    collective_detail::collective_local_validation(context.mesh, "Finalize composed raw source channels",
        [&]
        {
            result.result.sources = chemistry.sums.finalize(); // Exactly one narrowing after every stage.
            result.result.gross_events = std::move(chemistry.gross_events);
            result.chemistry_ledgers = std::move(chemistry.ledgers);
            result.post_faces = std::move(donors);
        });
    return result;
}

inline ComposedSourceStage compose_post_chemistry_faces(const Flow::SourceStageContext& context,
    const SpeciesCatalog& catalog, const LowMachFaceTransportAdapter& adapter,
    const std::function<void(const Flow::SourceStageContext&, LocalChemistryStage&)>& prepare_chemistry)
{
    const std::array<const LowMachFaceTransportAdapter*, 1> stages{&adapter};
    return compose_post_chemistry_faces(context, catalog, stages, prepare_chemistry);
}
} // namespace SimpleFluid::Reacting::low_mach_faces
