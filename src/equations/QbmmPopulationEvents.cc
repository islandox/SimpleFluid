#include "equations/details/QbmmPopulationEvents.hh"

#include "equations/QbmmBulkGasPool.hh"
#include "utils/CompensatedSum.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
using ExactStock = qbmm_face_detail::ScaledFaceSum;

struct JointStocks
{
    std::vector<ExactStock> original_remaining, credits;
    const QbmmBulkGasPoolBinding* pool;
    CellSpeciesState& proposed;
    qbmm_event_detail::RawSpeciesExactStocks* exact;
};

struct Rejection
{
    Status status;
    std::string diagnostic;
    std::optional<QbmmDonorShortfall> donor_shortfall;
};

struct StockShortfall
{
    long double available, missing;
};

std::optional<StockShortfall> reserve_stock(Sum& stock, double requested)
{
    const auto available = stock.value();
    stock += -static_cast<long double>(requested);
    if (!std::isfinite(stock.value()) || stock.value() < 0)
        return StockShortfall{available, -stock.value()};
    return std::nullopt;
}

std::optional<StockShortfall> reserve_stock(ExactStock& stock, double requested)
{
    stock.add_scaled_value(-requested, 0);
    if (stock.sign() >= 0)
        return std::nullopt;
    const double missing = -stock.rounded();
    stock.add_scaled_value(requested, 0);
    return StockShortfall{stock.rounded(), missing};
}

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

double narrow_exact_stock(const ExactStock& stock)
{
    const double rounded = stock.rounded();
    require(stock.sign() >= 0 && (stock.sign() == 0 || rounded > 0), Status::UnrepresentableEndpoint,
        "QBMM exact species stock cannot be represented as a nonnegative endpoint.");
    return narrow_nonnegative(rounded);
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

long double validate_transfer(const QbmmPopulationSchema& schema, const QbmmMaterialTransfer& transfer,
    long double signed_mass, const JointStocks* joint)
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
    require(positive(transfer_mass) && positive(std::abs(signed_mass)) && close(transfer_mass, std::abs(signed_mass)),
        Status::InvalidTransfer, "QBMM material transfer does not match the mass of its atom allocation.");
    if (const auto* phase = std::get_if<QbmmPhaseExchange>(&transfer.counterpart))
    {
        require(static_cast<std::size_t>(phase->phase) < static_cast<std::size_t>(SpeciesPhase::Count),
            Status::InvalidTransfer, "QBMM transfer has an invalid counterpart phase.");
        const auto& catalog = schema.catalog();
        for (std::size_t s = 0; s < transfer.moles.size(); ++s)
            require(transfer.moles[s] == 0 || catalog.allows(catalog.species()[s].id, phase->phase),
                Status::InvalidTransfer, "QBMM transfer species is disallowed in the counterpart phase.");
        if (joint && phase->phase == SpeciesPhase::Resolved)
        {
            const auto direction = signed_mass > 0 ? QbmmPoolConversionDirection::PoolToDispersed
                                                   : QbmmPoolConversionDirection::DispersedToPool;
            require(joint->pool && phase->resolved_pool &&
                        phase->resolved_pool->pool_fingerprint == joint->pool->fingerprint() &&
                        phase->resolved_pool->direction == direction,
                Status::InvalidTransfer, "QBMM Resolved conversion requires the bound pool and matching direction.");
            try
            {
                joint->pool->validate_family(schema, direction);
            }
            catch (const std::invalid_argument& error)
            {
                throw Rejection{Status::InvalidTransfer, error.what(), std::nullopt};
            }
        }
        else
            require(!phase->resolved_pool, Status::InvalidTransfer,
                "QBMM pool conversion provenance requires the joint Resolved event path.");
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
    const QbmmMaterialTransfer& transfer, long double signed_mass, Sum& material_mass, const JointStocks* joint)
{
    const auto transferred_mass = validate_transfer(schema, transfer, signed_mass, joint);
    const auto* phase = std::get_if<QbmmPhaseExchange>(&transfer.counterpart);
    require(!phase || !phase->resolved_pool || receipt.kind == QbmmEventKind::Birth ||
                receipt.kind == QbmmEventKind::Disappearance,
        Status::InvalidTransfer, "QBMM Resolved conversions require explicit birth or disappearance.");
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
    const QbmmEventBatch& batch, const QbmmPopulationEvent& event, bool adaptive, const JointStocks* joint)
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
            reserve_transfer(
                receipt, schema, allocation.transfer, atom_moments(allocation.atom)[1], material_mass, joint);
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
            (void) validate_transfer(schema, resize->transfer, -loss, joint);
            throw Rejection{Status::UnrepresentableEndpoint,
                "QBMM resize exceeds the donor size at this duration. Use explicit disappearance at zero.",
                std::nullopt};
        }
        require(positive(product.mass_over_scale) && product.mass_over_scale != donor.atom.mass_over_scale,
            Status::InvalidEvent,
            "QBMM resizing requires a distinct positive product mass. Use explicit disappearance at zero.");
        receipt.moment_change = resize_change(donor.atom, product);
        receipt.kind = receipt.moment_change[1] > 0 ? QbmmEventKind::Growth : QbmmEventKind::Shrinkage;
        reserve_transfer(receipt, schema, resize->transfer, receipt.moment_change[1], material_mass, joint);
        receipt.donors.push_back(donor);
        receipt.products.push_back(product);
    }
    else if (const auto* disappearance = std::get_if<QbmmDisappearance>(&event.operation))
    {
        const auto donor = donor_atom(initial, disappearance->donor, disappearance->count);
        receipt.kind = QbmmEventKind::Disappearance;
        reserve_transfer(receipt, schema, disappearance->transfer, -atom_moments(donor.atom)[1], material_mass, joint);
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

void add_count(Sum& sum, double count) { sum += count; }
void add_count(ExactStock& sum, double count) { sum.add_scaled_value(count, 0); }
long double count_value(const Sum& sum) { return sum.value(); }
double count_value(const ExactStock& sum) { return sum.rounded(); }

template<class Stock> void check_endpoint_changes(
    const QbmmQuadrature& pre, const QbmmEventMoments& constructive, const QbmmPopulationEndpoint& endpoint)
{
    MomentSums replay;
    std::array<Stock, 2> retained;
    for (std::size_t i = 0; i < pre.active_atoms; ++i)
        add_count(retained[i], pre.atoms[i].count);
    // Replay debits in count space before weighting them. Weighting two large
    // competing debits separately could obscure exact depletion of an atom.
    for (const auto& receipt : endpoint.receipts)
    {
        for (const auto& donor : receipt.donors)
            add_count(retained[donor.index], -donor.atom.count);
        for (const auto product : receipt.products)
            accumulate(replay, atom_moments(product));
    }
    for (std::size_t i = 0; i < pre.active_atoms; ++i)
        if (const auto count = count_value(retained[i]); count > 0)
            accumulate(replay, atom_moments(count, pre.atoms[i].mass_over_scale));
    const auto expected = finite_values(replay);
    for (std::size_t k = 0; k < expected.size(); ++k)
        require(close(expected[k], constructive[k]), Status::InconsistentEndpoint,
            "QBMM signed event moments do not reproduce the constructive endpoint.");
}
QbmmPopulationAdvance advance_population(const QbmmPopulationSchema& schema, const CellSpeciesState& initial,
    QbmmCellView pre_cell, const QbmmEventBatch& batch, std::optional<RawSpecies> raw_species,
    qbmm_event_detail::StageMomentChanges* adaptive_changes, JointStocks* joint = nullptr)
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

        QbmmPopulationEndpoint next{joint ? CellSpeciesState{} : initial, pre_cell.number, pre_cell.shape, pre, {}, {},
            {}, {}, schema.fingerprint()};
        if (batch.events.empty() && !raw_species)
            return {Status::Accepted, std::move(next), std::nullopt, {}, std::nullopt};
        auto stocks = joint ? std::vector<Sum>{} : make_stocks(initial, *family, raw_species);
        if (batch.events.empty())
        {
            if (adaptive_changes)
                *adaptive_changes = {};
            return {Status::Accepted, std::move(next), std::nullopt, {}, std::nullopt};
        }

        std::array<Sum, 2> remaining_count;
        std::array<ExactStock, 2> joint_counts;
        std::vector<ExactStock> joint_family;
        if (joint)
        {
            joint_family.resize(family->amounts.size());
            const auto offset =
                initial.amounts.size() + (family - initial.dispersedPopulations.begin()) * joint_family.size();
            for (std::size_t s = 0; s < joint_family.size(); ++s)
                if (joint->exact)
                    joint_family[s] = (*joint->exact)[offset + s];
                else
                    joint_family[s].add_scaled_value(family->amounts[s], 0);
        }
        for (std::size_t i = 0; i < pre.active_atoms; ++i)
            if (joint)
                joint_counts[i].add_scaled_value(pre.atoms[i].count, 0);
            else
                remaining_count[i] += pre.atoms[i].count;
        const auto family_offset = initial.amounts.size();
        bool checked_pool_withdrawal = false;
        next.receipts.reserve(batch.events.size());

        for (std::size_t i = 0; i < batch.events.size(); ++i)
        {
            failed_event = i;
            auto receipt = make_receipt(schema, pre, batch, batch.events[i], adaptive_changes != nullptr, joint);
            for (const auto& donor : receipt.donors)
            {
                const auto shortfall = joint ? reserve_stock(joint_counts[donor.index], donor.atom.count)
                                             : reserve_stock(remaining_count[donor.index], donor.atom.count);
                if (shortfall)
                    throw Rejection{Status::CountDonorOverdraw, "QBMM events exceed an original atom count donor.",
                        QbmmDonorShortfall{donor.index, shortfall->available, donor.atom.count, shortfall->missing}};
            }
            for (const auto& raw : receipt.transfers)
            {
                const auto donor = raw.into_family ? counterparty_offset(raw) : family_offset;
                if (donor)
                {
                    const auto* phase = std::get_if<QbmmPhaseExchange>(&raw.transfer.counterpart);
                    if (!checked_pool_withdrawal && raw.into_family && phase && phase->resolved_pool)
                    {
                        try
                        {
                            (void) schema.scaled_mass(
                                std::span{initial.amounts}.subspan(*donor, raw.transfer.moles.size()));
                        }
                        catch (const std::invalid_argument& error)
                        {
                            throw Rejection{Status::InvalidTransfer, error.what(), std::nullopt};
                        }
                        checked_pool_withdrawal = true;
                    }
                    for (std::size_t s = 0; s < raw.transfer.moles.size(); ++s)
                    {
                        const auto requested = raw.transfer.moles[s];
                        const auto shortfall = joint
                            ? reserve_stock(raw.into_family ? joint->original_remaining[*donor + s] : joint_family[s],
                                  requested)
                            : reserve_stock(stocks[*donor + s], requested);
                        if (shortfall)
                        {
                            const auto donor_phase = raw.into_family ? phase->phase : SpeciesPhase::Dispersed;
                            const QbmmSpeciesDonor species{schema.catalog().species()[s].id, donor_phase,
                                raw.into_family ? std::string{} : schema.population()};
                            throw Rejection{Status::SpeciesDonorOverdraw,
                                "QBMM transfers exceed an original species donor stock.",
                                QbmmDonorShortfall{species, shortfall->available, requested, shortfall->missing}};
                        }
                    }
                }
            }
            next.receipts.push_back(std::move(receipt));
        }
        failed_event.reset();

        MomentSums measure;
        for (std::size_t i = 0; i < pre.active_atoms; ++i)
        {
            const auto count = joint ? joint_counts[i].rounded() : narrow_nonnegative(remaining_count[i].value());
            if (count > 0)
                accumulate(measure, atom_moments({count, pre.atoms[i].mass_over_scale}));
        }
        MomentSums changes, consumed, created;
        Sum next_count;
        ExactStock joint_next_count;
        if (joint)
            add_count(joint_next_count, pre_cell.number);
        else
            add_count(next_count, pre_cell.number);
        for (const auto& receipt : next.receipts)
        {
            if (receipt.kind == QbmmEventKind::Birth || receipt.kind == QbmmEventKind::Breakup)
                append_atom_changes(changes, receipt, receipt.kind == QbmmEventKind::Breakup);
            else
                accumulate(changes, receipt.moment_change);
            accumulate(consumed, receipt.gross_consumed);
            accumulate(created, receipt.gross_created);
            for (const auto& donor : receipt.donors)
                if (joint)
                    add_count(joint_next_count, -donor.atom.count);
                else
                    add_count(next_count, -donor.atom.count);
            for (const auto product : receipt.products)
            {
                if (joint)
                    add_count(joint_next_count, product.count);
                else
                    add_count(next_count, product.count);
                accumulate(measure, atom_moments(product));
            }
            for (const auto& raw : receipt.transfers)
            {
                const auto receiver = raw.into_family ? family_offset : counterparty_offset(raw);
                if (receiver)
                    for (std::size_t s = 0; s < raw.transfer.moles.size(); ++s)
                    {
                        if (joint)
                        {
                            auto& stock = raw.into_family ? joint_family[s] : joint->credits[*receiver + s];
                            stock.add_scaled_value(raw.transfer.moles[s], 0);
                        }
                        else
                            stocks[*receiver + s] += raw.transfer.moles[s];
                    }
            }
        }
        // Evaluate H0 + delta by its well-conditioned form. If more than half of
        // H0 disappears, sum the surviving donor weights and positive products.
        // Subtracting rounded reconstructed donors from H0 can otherwise leave
        // the initial inversion residual larger than the entire surviving stock.
        // For small changes keep the compensated signed update from canonical H0.
        Sum count_change;
        if (measure[0].value() < .5L * pre_cell.number)
        {
            next_count = measure[0];
            count_change = next_count;
            count_change += -pre_cell.number;
        }
        else if (joint)
        {
            next_count += count_value(joint_next_count);
            // Subtract in the expansion before rounding. A resulting net change can be
            // representable even when adding it to the original count rounds away.
            add_count(joint_next_count, -pre_cell.number);
            const auto high = count_value(joint_next_count);
            count_change += high;
            add_count(joint_next_count, -high);
            count_change += count_value(joint_next_count);
        }
        else
        {
            count_change = next_count;
            count_change += -pre_cell.number;
        }
        next.moment_change = finite_values(changes);
        next.moment_change[0] = count_change.value();
        next.gross_consumed = finite_values(consumed);
        next.gross_created = finite_values(created);

        const auto constructive_raw = finite_values(measure);
        if (joint)
            check_endpoint_changes<ExactStock>(pre, constructive_raw, next);
        else
            check_endpoint_changes<Sum>(pre, constructive_raw, next);
        QbmmMomentBlock constructive;
        std::ranges::transform(constructive_raw, constructive.begin(), narrow_nonnegative);
        next.count = narrow_nonnegative(next_count.value());
        next.shape = {constructive[2], constructive[3]};
        auto& next_species = joint ? joint->proposed : next.species;
        auto& next_family = next_species.dispersedPopulations[family - initial.dispersedPopulations.begin()].amounts;
        if (!joint)
            for (std::size_t i = 0; i < next.species.amounts.size(); ++i)
                next.species.amounts[i] = narrow_nonnegative(stocks[i].value());
        for (std::size_t s = 0; s < next_family.size(); ++s)
            next_family[s] = joint ? narrow_exact_stock(joint_family[s])
                                   : narrow_nonnegative(stocks[family_offset + s].value());
        const QbmmCellView post_cell{schema.population(), next_species.schemaFingerprint, next_family, next.count,
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
        if (joint && joint->exact)
        {
            const auto offset = initial.amounts.size() +
                                (family - initial.dispersedPopulations.begin()) * joint_family.size();
            std::ranges::move(joint_family, joint->exact->begin() + offset);
        }
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

namespace
{
QbmmJointPopulationAdvance advance_populations(const CellSpeciesState& initial,
    std::span<const QbmmFamilyEventInput> families, const QbmmBulkGasPoolBinding* resolved_pool,
    std::optional<RawSpecies> raw, qbmm_event_detail::RawSpeciesExactStocks* exact_stocks,
    std::span<qbmm_event_detail::StageMomentChanges> adaptive_changes,
    std::span<std::array<long double, 4>> moment_changes)
{
    if (families.empty())
        return {Status::InvalidProvenance, std::nullopt, std::nullopt, std::nullopt,
            "QBMM joint advance requires at least one family.", std::nullopt};

    std::vector<std::size_t> order(families.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::stable_sort(order, {}, [&](std::size_t i) { return families[i].schema.population(); });
    for (std::size_t i = 1; i < order.size(); ++i)
        if (families[order[i - 1]].schema.population() == families[order[i]].schema.population())
            return {Status::InvalidProvenance, std::nullopt, order[i], std::nullopt,
                "QBMM joint advance requires distinct family identities.", std::nullopt};

    try
    {
        if (resolved_pool)
            resolved_pool->validate_inventory(initial);
        else
            initial.validate(families[order.front()].schema.catalog());
    }
    catch (const std::invalid_argument& error)
    {
        return {Status::InvalidState, std::nullopt, std::nullopt, std::nullopt, error.what(), std::nullopt};
    }

    const auto unnamed_size = initial.amounts.size();
    const auto raw_size =
        unnamed_size + initial.dispersedPopulations.size() * families.front().schema.catalog().species().size();
    if ((raw && raw->size() != raw_size) ||
        (exact_stocks && !exact_stocks->empty() && exact_stocks->size() != raw_size) ||
        (!adaptive_changes.empty() && adaptive_changes.size() != families.size()) ||
        (!moment_changes.empty() && moment_changes.size() != families.size()))
        return {Status::InvalidState, std::nullopt, std::nullopt, std::nullopt,
            "QBMM joint raw species or moment layout differs.", std::nullopt};

    qbmm_event_detail::RawSpeciesExactStocks exact;
    if (exact_stocks)
    {
        exact = *exact_stocks;
        if (exact.empty())
        {
            exact.resize(raw_size);
            for (std::size_t i = 0; i < raw_size; ++i)
                for (const long double part : (*raw)[i])
                    exact[i].add_unrounded_value(part);
        }
    }
    QbmmJointPopulationEndpoint next{initial, std::vector<QbmmFamilyEventEndpoint>(families.size())};
    JointStocks stocks{std::vector<ExactStock>(unnamed_size), std::vector<ExactStock>(unnamed_size), resolved_pool,
        next.species, exact_stocks ? &exact : nullptr};
    for (std::size_t i = 0; i < unnamed_size; ++i)
        if (exact_stocks)
            stocks.original_remaining[i] = exact[i];
        else
            stocks.original_remaining[i].add_scaled_value(initial.amounts[i], 0);

    const auto& stage = families[order.front()].batch;
    for (const auto i : order)
    {
        const auto& input = families[i];
        if (input.batch.substep != stage.substep || input.batch.duration != stage.duration)
            return {Status::InvalidProvenance, std::nullopt, i, std::nullopt,
                "QBMM joint families must share the same substep and duration.", std::nullopt};
        auto advance = advance_population(input.schema, initial, input.pre_cell, input.batch, std::nullopt,
            adaptive_changes.empty() ? nullptr : &adaptive_changes[i], &stocks);
        if (!advance.endpoint)
            return {advance.status, std::nullopt, i, advance.failed_event, std::move(advance.diagnostic),
                std::move(advance.rejected_event)};

        auto& family = *advance.endpoint;
        if (!moment_changes.empty())
            moment_changes[i] = family.moment_change;
        next.families[i] = {
            family.count, family.shape, family.quadrature, std::move(family.receipts), family.family_fingerprint};
    }

    try
    {
        for (std::size_t i = 0; i < unnamed_size; ++i)
        {
            auto& stock = stocks.original_remaining[i];
            stock.add_weighted(stocks.credits[i], 1.0);
            next.species.amounts[i] = narrow_exact_stock(stock);
            if (exact_stocks)
                exact[i] = std::move(stock);
        }
    }
    catch (const Rejection& error)
    {
        return {error.status, std::nullopt, std::nullopt, std::nullopt, error.diagnostic, std::nullopt};
    }
    if (exact_stocks)
        *exact_stocks = std::move(exact);
    return {Status::Accepted, std::move(next), std::nullopt, std::nullopt, {}, std::nullopt};
}

} // namespace

QbmmJointPopulationAdvance advance_qbmm_populations(const CellSpeciesState& initial,
    std::span<const QbmmFamilyEventInput> families, const QbmmBulkGasPoolBinding* resolved_pool)
{
    return advance_populations(initial, families, resolved_pool, std::nullopt, nullptr, {}, {});
}

QbmmJointPopulationAdvance qbmm_event_detail::advance_joint_with_raw_species(const CellSpeciesState& initial,
    std::span<const QbmmFamilyEventInput> families, const QbmmBulkGasPoolBinding& pool, RawSpecies raw_species,
    RawSpeciesExactStocks& exact_stocks, std::span<StageMomentChanges> adaptive_changes, std::span<std::array<long double, 4>> moment_changes)
{
    return advance_populations(initial, families, &pool, raw_species, &exact_stocks, adaptive_changes, moment_changes);
}

QbmmPopulationAdvance qbmm_event_detail::advance_with_raw_species(const QbmmPopulationSchema& schema,
    const CellSpeciesState& initial, QbmmCellView pre_cell, const QbmmEventBatch& batch, RawSpecies raw_species,
    StageMomentChanges* adaptive_changes)
{
    return advance_population(schema, initial, pre_cell, batch, raw_species, adaptive_changes);
}

} // namespace SimpleFluid::Reacting
