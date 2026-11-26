#include "equations/details/QbmmPopulationEvents.hh"

#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace SimpleFluid::Reacting
{
namespace
{
using Status = QbmmEventStatus;
using Sum = detail::CompensatedSum<long double>;
using MomentSums = std::array<Sum, 4>;

struct Rejection
{
    Status status;
    std::string diagnostic;
    std::optional<QbmmDonorShortfall> donor_shortfall;
};

void require(bool condition, Status status, std::string_view diagnostic)
{
    if (!condition)
        throw Rejection{status, std::string(diagnostic), std::nullopt};
}

bool positive(long double value)
{
    return std::isfinite(value) && value > 0;
}

bool close(long double first, long double second, long double tolerance = qbmm_reconstruction_tolerance)
{
    return std::isfinite(first) && std::isfinite(second) &&
           std::abs(first - second) <= tolerance * std::max(std::abs(first), std::abs(second));
}

double narrow_nonnegative(long double value)
{
    const auto narrowed = static_cast<double>(value);
    require(value >= 0 && (value == 0 || narrowed > 0) && close(value, narrowed), Status::UnrepresentableEndpoint,
        "QBMM endpoint or product cannot be represented as a nonnegative stock.");
    return narrowed;
}

QbmmEventMoments atom_moments(long double count, double mass_over_scale)
{
    require(positive(count) && positive(mass_over_scale), Status::InvalidEvent,
        "QBMM event atoms require finite positive count and mass.");
    QbmmEventMoments moments{count};
    for (std::size_t k = 1; k < moments.size(); ++k)
    {
        moments[k] = moments[k - 1] * mass_over_scale;
        require(
            positive(moments[k]), Status::UnrepresentableEndpoint, "QBMM event atom moments cannot be represented.");
    }
    return moments;
}

QbmmEventMoments atom_moments(QbmmAtom atom)
{
    return atom_moments(atom.count, atom.mass_over_scale);
}

void accumulate(MomentSums& sums, const QbmmEventMoments& terms)
{
    for (std::size_t k = 0; k < sums.size(); ++k)
        sums[k] += terms[k];
}

QbmmEventMoments finite_values(const MomentSums& sums)
{
    QbmmEventMoments result{};
    for (std::size_t k = 0; k < sums.size(); ++k)
    {
        result[k] = sums[k].value();
        require(std::isfinite(result[k]), Status::UnrepresentableEndpoint, "QBMM event moment sum is nonfinite.");
    }
    return result;
}

void append_atom_changes(MomentSums& sums, const QbmmEventReceipt& receipt, bool closed_mass = false)
{
    for (const auto& donor : receipt.donors)
    {
        const auto moments = atom_moments(donor.atom);
        for (std::size_t k = 0; k < sums.size(); ++k)
            if (k != 1 || !closed_mass)
                sums[k] += -moments[k];
    }
    for (const auto product : receipt.products)
    {
        const auto moments = atom_moments(product);
        for (std::size_t k = 0; k < sums.size(); ++k)
            if (k != 1 || !closed_mass)
                sums[k] += moments[k];
    }
}

QbmmDonorAtom donor_atom(const QbmmQuadrature& initial, std::size_t index, double count)
{
    require(index < initial.active_atoms && positive(count), Status::InvalidEvent,
        "QBMM event requires a valid pre-substep donor index and positive count.");
    return {index, {count, initial.atoms[index].mass_over_scale}};
}

long double validate_transfer(
    const QbmmPopulationSchema& schema, const QbmmMaterialTransfer& transfer, long double scaled_mass)
{
    double transfer_mass = 0;
    try
    {
        // Check composition on this allocation before any opposite transfer can cancel it.
        transfer_mass = schema.scaled_mass(transfer.moles);
    }
    catch (const std::invalid_argument& error)
    {
        throw Rejection{Status::InvalidTransfer, error.what(), std::nullopt};
    }
    require(positive(transfer_mass) && positive(scaled_mass) && close(transfer_mass, scaled_mass),
        Status::InvalidTransfer, "QBMM material transfer does not match the mass of its atom allocation.");
    if (const auto* phase = std::get_if<QbmmPhaseExchange>(&transfer.counterpart))
    {
        require(static_cast<std::size_t>(phase->phase) < static_cast<std::size_t>(SpeciesPhase::Count),
            Status::InvalidTransfer, "QBMM transfer has an invalid counterpart phase.");
        const auto& catalog = schema.catalog();
        for (std::size_t s = 0; s < transfer.moles.size(); ++s)
            require(transfer.moles[s] == 0 || catalog.allows(catalog.species()[s].id, phase->phase),
                Status::InvalidTransfer, "QBMM transfer species is disallowed in the counterpart phase.");
    }
    else
    {
        const auto& reservoir = std::get<QbmmReservoirExchange>(transfer.counterpart);
        require(!reservoir.reservoir.empty() && std::isfinite(reservoir.energy_j) &&
                    std::ranges::all_of(reservoir.momentum, [](double value) { return std::isfinite(value); }),
            Status::InvalidTransfer, "QBMM external transfer requires a reservoir and finite energy and momentum.");
    }
    return transfer_mass;
}

void reserve_transfer(QbmmEventReceipt& receipt, const QbmmPopulationSchema& schema,
    const QbmmMaterialTransfer& transfer, long double signed_mass, Sum& material_mass)
{
    const auto transferred_mass = validate_transfer(schema, transfer, std::abs(signed_mass));
    const bool into_family = signed_mass > 0;
    material_mass += into_family ? transferred_mass : -transferred_mass;
    receipt.transfers.push_back({transfer, into_family});
}

QbmmEventMoments resize_change(QbmmAtom donor, QbmmAtom product)
{
    const long double left = donor.mass_over_scale, right = product.mass_over_scale;
    const long double dH1 = (right - left) * donor.count;
    const auto left_term = dH1 * left;
    const auto right_term = dH1 * right;
    const auto dH2 = left_term + right_term;
    const auto dH3 = left_term * left + left_term * right + right_term * right;
    require(positive(std::abs(dH1)) && positive(std::abs(dH2)) && positive(std::abs(dH3)),
        Status::UnrepresentableEndpoint, "QBMM growth or shrinkage moment change cannot be represented.");
    return {0, dH1, dH2, dH3};
}

QbmmEventReceipt make_receipt(const QbmmPopulationSchema& schema, const QbmmQuadrature& initial,
    const QbmmEventBatch& batch, const QbmmPopulationEvent& event, bool adaptive)
{
    require(!event.provider.empty() && event.provider_revision > 0, Status::InvalidProvenance,
        "QBMM event requires a provider identity and positive revision.");
    QbmmEventReceipt receipt;
    receipt.population = schema.population();
    receipt.provider = event.provider;
    receipt.catalog_fingerprint = schema.catalog().fingerprint();
    receipt.family_fingerprint = schema.fingerprint();
    receipt.provider_revision = event.provider_revision;
    receipt.substep = batch.substep;
    receipt.duration = batch.duration;
    MomentSums direct_change;
    Sum material_mass;

    if (const auto* birth = std::get_if<QbmmBirth>(&event.operation))
    {
        require(!birth->atoms.empty(), Status::InvalidEvent, "QBMM birth requires an explicit positive distribution.");
        receipt.kind = QbmmEventKind::Birth;
        for (const auto& allocation : birth->atoms)
        {
            reserve_transfer(receipt, schema, allocation.transfer, atom_moments(allocation.atom)[1], material_mass);
            receipt.products.push_back(allocation.atom);
        }
    }
    else if (const auto* resize = std::get_if<QbmmResize>(&event.operation))
    {
        const auto donor = donor_atom(initial, resize->donor, resize->count);
        const QbmmAtom product{resize->count, resize->mass_over_scale};
        if (adaptive && std::isfinite(product.mass_over_scale) && product.mass_over_scale <= 0)
        {
            const long double loss =
                (static_cast<long double>(donor.atom.mass_over_scale) - product.mass_over_scale) * donor.atom.count;
            (void) validate_transfer(schema, resize->transfer, loss);
            throw Rejection{Status::UnrepresentableEndpoint,
                "QBMM resize exceeds the donor size at this duration. Use explicit disappearance at zero.",
                std::nullopt};
        }
        require(positive(product.mass_over_scale) && product.mass_over_scale != donor.atom.mass_over_scale,
            Status::InvalidEvent,
            "QBMM resizing requires a distinct positive product mass. Use explicit disappearance at zero.");
        receipt.moment_change = resize_change(donor.atom, product);
        receipt.kind = receipt.moment_change[1] > 0 ? QbmmEventKind::Growth : QbmmEventKind::Shrinkage;
        reserve_transfer(receipt, schema, resize->transfer, receipt.moment_change[1], material_mass);
        receipt.donors.push_back(donor);
        receipt.products.push_back(product);
    }
    else if (const auto* disappearance = std::get_if<QbmmDisappearance>(&event.operation))
    {
        const auto donor = donor_atom(initial, disappearance->donor, disappearance->count);
        receipt.kind = QbmmEventKind::Disappearance;
        reserve_transfer(receipt, schema, disappearance->transfer, -atom_moments(donor.atom)[1], material_mass);
        receipt.donors.push_back(donor);
    }
    else if (const auto* coalescence = std::get_if<QbmmCoalescence>(&event.operation))
    {
        const auto first = donor_atom(initial, coalescence->first, coalescence->count);
        const auto second = donor_atom(initial, coalescence->second, coalescence->count);
        const long double a = first.atom.mass_over_scale, b = second.atom.mass_over_scale;
        const double mass = narrow_nonnegative(a + b);
        require(mass > std::max(a, b), Status::UnrepresentableEndpoint,
            "QBMM coalescence cannot represent a product larger than both of its parents.");
        const long double pair_mass = (coalescence->count * a) * b;
        const long double dH2 = 2 * pair_mass;
        const long double dH3 = 3 * ((pair_mass * a) + (pair_mass * b));
        require(positive(dH2) && positive(dH3), Status::UnrepresentableEndpoint,
            "QBMM coalescence moment change cannot be represented.");
        receipt.kind = QbmmEventKind::Coalescence;
        receipt.donors = {first, second};
        receipt.products.push_back({coalescence->count, mass});
        receipt.moment_change = {-coalescence->count, 0, dH2, dH3};
    }
    else
    {
        const auto& breakup = std::get<QbmmBreakup>(event.operation);
        const auto donor = donor_atom(initial, breakup.donor, breakup.count);
        MomentSums per_parent;
        for (const auto daughter : breakup.daughters)
        {
            require(daughter.mass_over_scale <= donor.atom.mass_over_scale, Status::InvalidEvent,
                "QBMM breakup daughter cannot be heavier than its parent.");
            accumulate(per_parent, atom_moments(daughter));
            const long double count = static_cast<long double>(breakup.count) * daughter.count;
            require(positive(count), Status::UnrepresentableEndpoint, "QBMM daughter count cannot be represented.");
            receipt.products.push_back({narrow_nonnegative(count), daughter.mass_over_scale});
        }
        const auto daughter_moments = finite_values(per_parent);
        require(daughter_moments[0] >= 1 && close(daughter_moments[1], donor.atom.mass_over_scale),
            Status::InvalidEvent,
            "QBMM breakup requires at least one expected daughter and must conserve parent mass.");
        receipt.kind = QbmmEventKind::Breakup;
        receipt.donors.push_back(donor);
        const auto donor_moments = atom_moments({1, donor.atom.mass_over_scale});
        for (std::size_t k = 0; k < direct_change.size(); ++k)
            direct_change[k] += -donor_moments[k];
        for (const auto daughter : breakup.daughters)
            accumulate(direct_change, atom_moments(daughter));
        receipt.moment_change = finite_values(direct_change);
        for (auto& change : receipt.moment_change)
            change *= breakup.count;
        receipt.moment_change[1] = 0; // Closed topology, checked against the daughter allocation above.
    }

    MomentSums consumed, created;
    for (const auto& donor : receipt.donors)
        accumulate(consumed, atom_moments(donor.atom));
    for (const auto product : receipt.products)
        accumulate(created, atom_moments(product));
    receipt.gross_consumed = finite_values(consumed);
    receipt.gross_created = finite_values(created);
    if (receipt.kind == QbmmEventKind::Birth)
        receipt.moment_change = receipt.gross_created;
    if (receipt.kind == QbmmEventKind::Disappearance)
        for (std::size_t k = 0; k < receipt.moment_change.size(); ++k)
            receipt.moment_change[k] = -receipt.gross_consumed[k];
    receipt.material_mass_change_over_scale = material_mass.value();
    require(close(receipt.moment_change[1], receipt.material_mass_change_over_scale), Status::InvalidTransfer,
        "QBMM signed material receipt and atom mass change disagree.");
    MomentSums raw_change;
    append_atom_changes(raw_change, receipt);
    const auto represented_change = finite_values(raw_change);
    for (std::size_t k = 0; k < represented_change.size(); ++k)
        require(std::abs(represented_change[k] - receipt.moment_change[k]) <=
                    qbmm_reconstruction_tolerance * std::max(receipt.gross_consumed[k], receipt.gross_created[k]),
            Status::InconsistentEndpoint, "QBMM signed event algebra disagrees with the supplied atom allocations.");
    return receipt;
}

using RawSpecies = std::span<const std::array<long double, 2>>;

std::vector<Sum> make_stocks(
    const CellSpeciesState& initial, const SpeciesPopulationState& family, std::optional<RawSpecies> raw)
{
    const auto& base = initial.amounts;
    std::vector<Sum> result(base.size() + family.amounts.size());
    if (!raw)
    {
        for (std::size_t i = 0; i < base.size(); ++i)
            result[i] += base[i];
        for (std::size_t s = 0; s < family.amounts.size(); ++s)
            result[base.size() + s] += family.amounts[s];
        return result;
    }

    std::size_t index = 0;
    const auto seed = [&](double represented)
    {
        require(index < raw->size(), Status::InvalidState, "QBMM raw species predecessor layout differs.");
        const auto& parts = (*raw)[index++];
        Sum stock;
        stock += parts[0];
        stock += parts[1];
        const auto value = stock.value();
        require(std::isfinite(parts[0]) && std::isfinite(parts[1]) && std::isfinite(value) && value >= 0 &&
                    (value == 0) == (represented == 0) && static_cast<double>(value) == represented,
            Status::InvalidState, "QBMM raw species predecessor does not represent its original inventory.");
        return stock;
    };
    for (std::size_t i = 0; i < base.size(); ++i)
        result[i] = seed(base[i]);
    for (const auto& population : initial.dispersedPopulations)
        for (std::size_t s = 0; s < population.amounts.size(); ++s)
        {
            auto stock = seed(population.amounts[s]);
            if (&population == &family)
                result[base.size() + s] = stock;
        }
    require(index == raw->size(), Status::InvalidState, "QBMM raw species predecessor layout differs.");
    return result;
}

std::optional<std::size_t> counterparty_offset(const QbmmMaterialTransferReceipt& raw)
{
    if (const auto* other = std::get_if<QbmmPhaseExchange>(&raw.transfer.counterpart))
        return static_cast<std::size_t>(other->phase) * raw.transfer.moles.size();
    return std::nullopt;
}

QbmmMomentBlock cell_moments(const QbmmPopulationSchema& schema, QbmmCellView cell, Status failure)
{
    try
    {
        return qbmm_cell_moments(schema, cell);
    }
    catch (const std::invalid_argument& error)
    {
        throw Rejection{failure, error.what(), std::nullopt};
    }
}

void check_endpoint_changes(
    const QbmmQuadrature& pre, const QbmmEventMoments& constructive, const QbmmPopulationEndpoint& endpoint)
{
    MomentSums replay;
    std::array<Sum, 2> retained;
    for (std::size_t i = 0; i < pre.active_atoms; ++i)
        retained[i] += pre.atoms[i].count;
    // Replay debits in count space before weighting them. Weighting two large
    // competing debits separately could obscure exact depletion of an atom.
    for (const auto& receipt : endpoint.receipts)
    {
        for (const auto& donor : receipt.donors)
            retained[donor.index] += -donor.atom.count;
        for (const auto product : receipt.products)
            accumulate(replay, atom_moments(product));
    }
    for (std::size_t i = 0; i < pre.active_atoms; ++i)
        if (retained[i].value() > 0)
            accumulate(replay, atom_moments(retained[i].value(), pre.atoms[i].mass_over_scale));
    const auto expected = finite_values(replay);
    for (std::size_t k = 0; k < expected.size(); ++k)
        require(close(expected[k], constructive[k]), Status::InconsistentEndpoint,
            "QBMM signed event moments do not reproduce the constructive endpoint.");
}
QbmmPopulationAdvance advance_population(const QbmmPopulationSchema& schema, const CellSpeciesState& initial,
    QbmmCellView pre_cell, const QbmmEventBatch& batch, std::optional<RawSpecies> raw_species,
    qbmm_event_detail::StageMomentChanges* adaptive_changes)
{
    std::optional<std::size_t> failed_event;
    try
    {
        try
        {
            initial.validate(schema.catalog());
        }
        catch (const std::invalid_argument& error)
        {
            throw Rejection{Status::InvalidState, error.what(), std::nullopt};
        }
        const auto family =
            std::ranges::find(initial.dispersedPopulations, schema.population(), &SpeciesPopulationState::id);
        require(family != initial.dispersedPopulations.end(), Status::InvalidState,
            "QBMM species state is missing the schema's named dispersed family.");
        require(pre_cell.species_moles.data() == family->amounts.data() &&
                    pre_cell.species_moles.size() == family->amounts.size(),
            Status::InvalidState, "QBMM initial cell view must alias the named family's authoritative species row.");
        const auto canonical_pre = cell_moments(schema, pre_cell, Status::InvalidState);
        const auto pre = invert_qbmm_moments(canonical_pre);
        if (!pre.valid())
            throw Rejection{Status::InvalidState,
                "QBMM predecessor '" + schema.population() + "' failed inversion (" +
                    qbmm_inversion_status_name(pre.status) + ").",
                std::nullopt};
        require(batch.family_fingerprint == schema.fingerprint() && std::isfinite(batch.duration) &&
                    batch.duration >= 0 && (batch.events.empty() || batch.duration > 0),
            Status::InvalidProvenance,
            "QBMM batch requires the matching family fingerprint and valid substep duration.");

        QbmmPopulationEndpoint next{
            initial, pre_cell.number, pre_cell.shape, pre, {}, {}, {}, {}, schema.fingerprint()};
        if (batch.events.empty() && !raw_species)
            return {Status::Accepted, std::move(next), std::nullopt, {}, std::nullopt};
        auto stocks = make_stocks(initial, *family, raw_species);
        if (batch.events.empty())
        {
            if (adaptive_changes)
                *adaptive_changes = {};
            return {Status::Accepted, std::move(next), std::nullopt, {}, std::nullopt};
        }

        std::array<Sum, 2> remaining_count;
        for (std::size_t i = 0; i < pre.active_atoms; ++i)
            remaining_count[i] += pre.atoms[i].count;
        const auto family_offset = initial.amounts.size();
        next.receipts.reserve(batch.events.size());

        for (std::size_t i = 0; i < batch.events.size(); ++i)
        {
            failed_event = i;
            auto receipt = make_receipt(schema, pre, batch, batch.events[i], adaptive_changes != nullptr);
            for (const auto& donor : receipt.donors)
            {
                auto& stock = remaining_count[donor.index];
                const auto available = stock.value();
                stock += -static_cast<long double>(donor.atom.count);
                if (!std::isfinite(stock.value()) || stock.value() < 0)
                {
                    const auto shortfall = -stock.value();
                    throw Rejection{Status::CountDonorOverdraw, "QBMM events exceed an original atom count donor.",
                        QbmmDonorShortfall{donor.index, available, donor.atom.count, shortfall}};
                }
            }
            for (const auto& raw : receipt.transfers)
            {
                const auto donor = raw.into_family ? counterparty_offset(raw) : family_offset;
                if (donor)
                {
                    for (std::size_t s = 0; s < raw.transfer.moles.size(); ++s)
                    {
                        const auto requested = raw.transfer.moles[s];
                        auto& stock = stocks[*donor + s];
                        const auto available = stock.value();
                        stock += -static_cast<long double>(requested);
                        if (!std::isfinite(stock.value()) || stock.value() < 0)
                        {
                            const auto shortfall = -stock.value();
                            const auto phase = raw.into_family
                                                   ? std::get<QbmmPhaseExchange>(raw.transfer.counterpart).phase
                                                   : SpeciesPhase::Dispersed;
                            const QbmmSpeciesDonor species{schema.catalog().species()[s].id, phase,
                                raw.into_family ? std::string{} : schema.population()};
                            throw Rejection{Status::SpeciesDonorOverdraw,
                                "QBMM transfers exceed an original species donor stock.",
                                QbmmDonorShortfall{species, available, requested, shortfall}};
                        }
                    }
                }
            }
            next.receipts.push_back(std::move(receipt));
        }
        failed_event.reset();

        MomentSums measure;
        for (std::size_t i = 0; i < pre.active_atoms; ++i)
            if (remaining_count[i].value() > 0)
                accumulate(measure,
                    atom_moments({narrow_nonnegative(remaining_count[i].value()), pre.atoms[i].mass_over_scale}));
        MomentSums changes, consumed, created;
        Sum next_count;
        next_count += pre_cell.number;
        for (const auto& receipt : next.receipts)
        {
            if (receipt.kind == QbmmEventKind::Birth || receipt.kind == QbmmEventKind::Breakup)
                append_atom_changes(changes, receipt, receipt.kind == QbmmEventKind::Breakup);
            else
                accumulate(changes, receipt.moment_change);
            accumulate(consumed, receipt.gross_consumed);
            accumulate(created, receipt.gross_created);
            for (const auto& donor : receipt.donors)
                next_count += -donor.atom.count;
            for (const auto product : receipt.products)
            {
                next_count += product.count;
                accumulate(measure, atom_moments(product));
            }
            for (const auto& raw : receipt.transfers)
            {
                const auto receiver = raw.into_family ? family_offset : counterparty_offset(raw);
                if (receiver)
                    for (std::size_t s = 0; s < raw.transfer.moles.size(); ++s)
                        stocks[*receiver + s] += raw.transfer.moles[s];
            }
        }
        // Evaluate H0 + delta by its well-conditioned form. If more than half of
        // H0 disappears, sum the surviving donor weights and positive products.
        // Subtracting rounded reconstructed donors from H0 can otherwise leave
        // the initial inversion residual larger than the entire surviving stock.
        // For small changes keep the compensated signed update from canonical H0.
        if (measure[0].value() < .5L * pre_cell.number)
            next_count = measure[0];
        next.moment_change = finite_values(changes);
        auto count_change = next_count;
        count_change += -pre_cell.number;
        next.moment_change[0] = count_change.value();
        next.gross_consumed = finite_values(consumed);
        next.gross_created = finite_values(created);

        const auto constructive_raw = finite_values(measure);
        check_endpoint_changes(pre, constructive_raw, next);
        QbmmMomentBlock constructive;
        std::ranges::transform(constructive_raw, constructive.begin(), narrow_nonnegative);
        next.count = narrow_nonnegative(next_count.value());
        next.shape = {constructive[2], constructive[3]};
        auto& next_family = next.species.dispersedPopulations[family - initial.dispersedPopulations.begin()].amounts;
        for (std::size_t i = 0; i < next.species.amounts.size(); ++i)
            next.species.amounts[i] = narrow_nonnegative(stocks[i].value());
        for (std::size_t s = 0; s < next_family.size(); ++s)
            next_family[s] = narrow_nonnegative(stocks[family_offset + s].value());
        const QbmmCellView post_cell{schema.population(), next.species.schemaFingerprint, next_family, next.count,
            next.shape, next.family_fingerprint};
        const auto post = cell_moments(schema, post_cell, Status::InconsistentEndpoint);
        for (std::size_t k = 0; k < post.size(); ++k)
            require((post[k] == 0) == (constructive[k] == 0) && close(post[k], constructive[k]),
                Status::InconsistentEndpoint,
                "QBMM constructive moments disagree with authoritative species or count.");
        next.quadrature = invert_qbmm_moments(post);
        if (!next.quadrature.valid())
            throw Rejection{Status::InconsistentEndpoint,
                "QBMM endpoint '" + schema.population() + "' failed inversion (" +
                    qbmm_inversion_status_name(next.quadrature.status) + ").",
                std::nullopt};
        if (adaptive_changes)
            *adaptive_changes = {count_change, {changes[2], changes[3]}};
        return {Status::Accepted, std::move(next), std::nullopt, {}, std::nullopt};
    }
    catch (const Rejection& error)
    {
        std::optional<QbmmRejectedEventReceipt> rejected;
        if (failed_event)
            rejected =
                QbmmRejectedEventReceipt{schema.population(), batch.family_fingerprint, schema.catalog().fingerprint(),
                    batch.substep, batch.duration, batch.events[*failed_event], error.status, error.donor_shortfall};
        return {error.status, std::nullopt, failed_event, error.diagnostic, std::move(rejected)};
    }
}
} // namespace

QbmmPopulationAdvance advance_qbmm_population(const QbmmPopulationSchema& schema, const CellSpeciesState& initial,
    QbmmCellView pre_cell, const QbmmEventBatch& batch)
{
    return advance_population(schema, initial, pre_cell, batch, std::nullopt, nullptr);
}

QbmmPopulationAdvance qbmm_event_detail::advance_with_raw_species(const QbmmPopulationSchema& schema,
    const CellSpeciesState& initial, QbmmCellView pre_cell, const QbmmEventBatch& batch, RawSpecies raw_species,
    StageMomentChanges* adaptive_changes)
{
    return advance_population(schema, initial, pre_cell, batch, raw_species, adaptive_changes);
}

} // namespace SimpleFluid::Reacting
