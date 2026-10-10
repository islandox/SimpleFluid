#include "equations/LumpedInnerCavityCoupling.hh"

#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

namespace SimpleFluid
{
namespace
{
template<class Operation> void collective(const Teuchos::Comm<int>& comm, Operation&& operation)
{
    std::exception_ptr error;
    try
    {
        operation();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const int failed = error ? 1 : 0;
    int any = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any);
    if (error)
        std::rethrow_exception(error);
    if (any)
        throw std::runtime_error("Cavity field operation failed on another rank.");
}

void agree(const Teuchos::Comm<int>& comm, const std::string& local)
{
    collective(comm,
        [&]
        {
            if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Cavity configuration exceeds collective count range.");
        });
    int count = static_cast<int>(local.size());
    Teuchos::broadcast(comm, 0, 1, &count);
    auto root = comm.getRank() == 0 ? local : std::string(count, '\0');
    if (count)
        Teuchos::broadcast(comm, 0, count, root.data());
    collective(comm,
        [&]
        {
            if (root != local)
                throw std::invalid_argument("Cavity field configuration or result differs between ranks.");
        });
}

void write_exchange(std::ostream& s, const LumpedInnerCavityExchange& exchange)
{
    s << exchange.air_energy_rate << ' ' << exchange.vapor_mass_rate << ' ' << exchange.surfaces.size() << ' ';
    for (const auto& e : exchange.surfaces)
        s << e.heat_transfer_coefficient << ' ' << e.vapor_mass_rate << ' ' << e.sensible_heat_rate << ' '
          << e.latent_heat_rate << ' ' << e.radiation_heat_rate << ' ' << e.heat_rate << ' ' << e.energy_rate << ' ';
}
} // namespace

InnerCavityFieldInput::InnerCavityFieldInput(
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::vector<InnerCavityFieldBatch> batches)
    : d_comm(std::move(comm)), d_batches(std::move(batches))
{
    if (d_comm.is_null())
        throw std::invalid_argument("Cavity field input requires a communicator.");
}

void InnerCavityFieldInput::configure() const
{
    const int local_needed = !d_configured;
    int needed = 0;
    Teuchos::reduceAll(*d_comm, Teuchos::REDUCE_MAX, 1, &local_needed, &needed);
    if (!needed)
        return;
    std::ostringstream signature;
    collective(*d_comm,
        [&]
        {
            if (d_batches.empty() || d_batches.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Cavity input requires a nonempty, representable batch count.");
            signature << std::setprecision(17) << d_batches.size() << ' ';
            std::set<std::string> names;
            std::set<std::pair<const void*, int>> batches;
            for (const auto& b : d_batches)
            {
                if (!b.d_mesh || !b.d_read || b.d_comm.is_null() || !communicators_congruent(*b.d_comm, *d_comm) ||
                    b.d_properties.name.empty() || !names.insert(b.d_properties.name).second ||
                    !batches.emplace(b.d_mesh.get(), b.d_batch_id).second ||
                    (b.d_sampling != InnerCavitySampling::Facewise && b.d_sampling != InnerCavitySampling::AreaAverage))
                    throw std::invalid_argument("Invalid/duplicate cavity batch or incompatible communicator.");
                const auto& p = b.d_properties;
                signature << std::quoted(p.name) << ' ' << b.d_batch_id << ' ' << static_cast<int>(b.d_sampling) << ' '
                          << static_cast<int>(p.moisture_exchange) << ' ' << p.mass_transfer_coefficient << ' '
                          << p.emissivity << ' ' << bool(p.convection_coefficient) << ' '
                          << convection_detail::configuration_signature({{"surface", p.convection}});
            }
        });
    agree(*d_comm, signature.str());
    collective(*d_comm, [&] { d_plans.resize(d_batches.size()); });
    d_configured = true;
}

void InnerCavityFieldInput::rebuild_plan(
    std::size_t batch, const std::vector<InnerCavityFieldBatch::FaceEntry>& local) const
{
    BatchPlan candidate;
    collective(*d_comm,
        [&]
        {
            for (const auto& f : local)
                candidate.local_ids.push_back(f.face);
            candidate.local_to_sorted.resize(local.size(), 0);
        });
    if (d_batches[batch].d_sampling == InnerCavitySampling::AreaAverage)
    {
        // Mesh ownership is authoritative. No global face table is needed for
        // an average; keep just local face IDs for topology-change detection.
        const global_index_t count = local.size();
        global_index_t total = 0;
        Teuchos::reduceAll(*d_comm, Teuchos::REDUCE_SUM, 1, &count, &total);
        if (total == 0)
            throw std::invalid_argument("Unknown or globally empty cavity batch.");
    }
    else
    {
        const int count = static_cast<int>(local.size());
        collective(*d_comm, [&] { candidate.rank_counts.resize(d_comm->getSize()); });
        Teuchos::gatherAll(*d_comm, 1, &count, d_comm->getSize(), candidate.rank_counts.data());
        std::vector<global_index_t> gathered;
        std::size_t total = 0;
        collective(*d_comm,
            [&]
            {
                for (int n : candidate.rank_counts)
                    total += n;
                if (total == 0 || total > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
                    throw std::invalid_argument("Empty or oversized cavity facewise batch.");
                gathered.resize(total);
            });
        std::size_t offset = 0, local_offset = 0;
        for (int rank = 0; rank < d_comm->getSize(); ++rank)
        {
            const int n = candidate.rank_counts[rank];
            if (rank == d_comm->getRank())
            {
                local_offset = offset;
                std::copy(candidate.local_ids.begin(), candidate.local_ids.end(), gathered.begin() + offset);
            }
            if (n)
                Teuchos::broadcast(*d_comm, rank, n, gathered.data() + offset);
            offset += n;
        }
        collective(*d_comm,
            [&]
            {
                candidate.global_ids = gathered;
                std::sort(candidate.global_ids.begin(), candidate.global_ids.end());
                if (std::adjacent_find(candidate.global_ids.begin(), candidate.global_ids.end()) !=
                    candidate.global_ids.end())
                    throw std::invalid_argument("Cavity face has multiple owning ranks.");
                candidate.gathered_to_sorted.reserve(total);
                for (auto id : gathered)
                    candidate.gathered_to_sorted.push_back(
                        std::lower_bound(candidate.global_ids.begin(), candidate.global_ids.end(), id) -
                        candidate.global_ids.begin());
                for (std::size_t i = 0; i < local.size(); ++i)
                    candidate.local_to_sorted[i] = candidate.gathered_to_sorted[local_offset + i];
            });
    }
    candidate.ready = true;
    d_plans[batch] = std::move(candidate);
}

auto InnerCavityFieldInput::sample() const -> Snapshot
{
    configure();
    std::vector<std::vector<InnerCavityFieldBatch::FaceEntry>> local;
    std::vector<int> local_changed, changed;
    collective(*d_comm,
        [&]
        {
            local.resize(d_batches.size());
            local_changed.resize(d_batches.size());
            changed.resize(d_batches.size());
            for (std::size_t i = 0; i < d_batches.size(); ++i)
            {
                const auto& batch = d_batches[i];
                local[i] = batch.d_read(batch.d_mesh.get(), batch.d_batch_id);
                if (local[i].size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
                    throw std::length_error("Cavity batch exceeds collective count range.");
                for (const auto& f : local[i])
                    if (!std::isfinite(f.area) || f.area <= 0.0 || !std::isfinite(f.temperature) ||
                        f.temperature <= 0.0)
                        throw std::invalid_argument("Invalid or repeated cavity face sample.");
                const auto& plan = d_plans[i];
                local_changed[i] = !plan.ready || plan.local_ids.size() != local[i].size();
                if (!local_changed[i])
                    for (std::size_t j = 0; j < local[i].size(); ++j)
                        local_changed[i] |= plan.local_ids[j] != local[i][j].face;
            }
        });
    Teuchos::reduceAll(
        *d_comm, Teuchos::REDUCE_MAX, static_cast<int>(changed.size()), local_changed.data(), changed.data());
    if (std::any_of(changed.begin(), changed.end(), [](int value) { return value != 0; }))
        collective(*d_comm,
            [&]
            {
                std::set<std::pair<const void*, global_index_t>> seen;
                for (std::size_t i = 0; i < local.size(); ++i)
                    for (const auto& f : local[i])
                        if (!seen.emplace(d_batches[i].d_mesh.get(), f.face).second)
                            throw std::invalid_argument("Repeated cavity boundary face.");
            });
    for (std::size_t i = 0; i < changed.size(); ++i)
        if (changed[i])
            rebuild_plan(i, local[i]);

    Snapshot result;
    for (std::size_t i = 0; i < d_batches.size(); ++i)
    {
        const auto& batch = d_batches[i];
        const auto& plan = d_plans[i];
        const auto offset = result.surfaces.size();
        if (batch.d_sampling == InnerCavitySampling::AreaAverage)
        {
            real_t sums[2]{}, totals[2]{};
            collective(*d_comm,
                [&]
                {
                    for (const auto& face : local[i])
                    {
                        sums[0] += face.area;
                        sums[1] += face.area * face.temperature;
                    }
                    if (!std::isfinite(sums[0]) || !std::isfinite(sums[1]))
                        throw std::overflow_error("Cavity patch average overflow.");
                });
            Teuchos::reduceAll(*d_comm, Teuchos::REDUCE_SUM, 2, sums, totals);
            collective(*d_comm,
                [&]
                {
                    if (!std::isfinite(totals[0]) || !std::isfinite(totals[1]) || totals[0] <= 0.0)
                        throw std::overflow_error("Invalid global cavity patch average.");
                    result.surfaces.push_back({&batch.d_properties, totals[0], totals[1] / totals[0]});
                });
        }
        else
        {
            std::vector<real_t> values;
            collective(*d_comm,
                [&]
                {
                    values.resize(2 * *std::max_element(plan.rank_counts.begin(), plan.rank_counts.end()));
                    result.surfaces.resize(offset + plan.global_ids.size());
                });
            std::size_t gathered_offset = 0;
            for (int rank = 0; rank < d_comm->getSize(); ++rank)
            {
                const int count = plan.rank_counts[rank];
                if (rank == d_comm->getRank())
                    for (int j = 0; j < count; ++j)
                    {
                        values[2 * j] = local[i][j].area;
                        values[2 * j + 1] = local[i][j].temperature;
                    }
                if (count)
                    Teuchos::broadcast(*d_comm, rank, 2 * count, values.data());
                for (int j = 0; j < count; ++j)
                    result.surfaces[offset + plan.gathered_to_sorted[gathered_offset + j]] = {
                        &batch.d_properties, values[2 * j], values[2 * j + 1]};
                gathered_offset += count;
            }
        }
        collective(*d_comm,
            [&]
            {
                result.boundaries.push_back({batch.d_mesh, batch.d_batch_id, {}});
                result.indices.emplace_back();
                auto& faces = result.boundaries.back().faces;
                faces.reserve(local[i].size());
                for (std::size_t j = 0; j < local[i].size(); ++j)
                {
                    const auto& f = local[i][j];
                    faces.push_back({f.face, f.slot, f.area, {}});
                    result.indices.back().push_back(offset + plan.local_to_sorted[j]);
                }
            });
    }
    return result;
}

void InnerCavityFieldInput::attach(Snapshot&& snapshot, LumpedInnerCavityExchange& exchange) const
{
    collective(*d_comm,
        [&]
        {
            exchange.boundaries = std::move(snapshot.boundaries);
            for (std::size_t i = 0; i < exchange.boundaries.size(); ++i)
                for (std::size_t j = 0; j < exchange.boundaries[i].faces.size(); ++j)
                {
                    auto& face = exchange.boundaries[i].faces[j];
                    const auto index = snapshot.indices[i][j];
                    face.exchange = exchange.surfaces[index];
                    const auto fraction = face.area / snapshot.surfaces[index].area;
                    face.exchange.vapor_mass_rate *= fraction;
                    face.exchange.sensible_heat_rate *= fraction;
                    face.exchange.latent_heat_rate *= fraction;
                    face.exchange.radiation_heat_rate *= fraction;
                    face.exchange.heat_rate *= fraction;
                    face.exchange.energy_rate *= fraction;
                }
        });
}

void InnerCavityFieldInput::validate_model(
    const LumpedInnerCavityModel& model, real_t dt, std::span<const real_t> factors) const
{
    std::ostringstream s;
    s << std::setprecision(17);
    const auto& o = model.options();
    s << model.state().temperature << ' ' << model.state().humidity_ratio << ' ' << dt << ' ' << o.volume << ' '
      << o.dry_air_mass << ' ' << o.dry_air_gas_constant << ' ' << o.vapor_gas_constant << ' ' << o.dry_air_cp << ' '
      << o.vapor_cp << ' ' << o.liquid_cp << ' ' << o.reference_temperature << ' ' << o.reference_latent_heat << ' '
      << o.minimum_temperature << ' ' << o.maximum_temperature << ' ' << factors.size() << ' ';
    for (auto f : factors)
        s << f << ' ';
    agree(*d_comm, s.str());
}

void InnerCavityFieldInput::validate_result(const LumpedInnerCavityStep& step) const
{
    std::ostringstream s;
    s << std::setprecision(17) << step.state.temperature << ' ' << step.state.humidity_ratio << ' '
      << step.mass_balance_residual << ' ' << step.energy_balance_residual << ' ';
    write_exchange(s, step.exchange);
    agree(*d_comm, s.str());
}

LumpedInnerCavityExchange InnerCavityFieldInput::evaluate(
    const LumpedInnerCavityModel& model, std::span<const real_t> factors) const
{
    validate_model(model, 0.0, factors);
    auto snapshot = sample();
    LumpedInnerCavityStep result;
    result.state = model.state();
    collective(*d_comm, [&] { result.exchange = model.evaluate_samples(snapshot.surfaces, factors); });
    validate_result(result);
    attach(std::move(snapshot), result.exchange);
    return result.exchange;
}

LumpedInnerCavityStep InnerCavityFieldInput::preview(
    const LumpedInnerCavityModel& model, real_t dt, std::span<const real_t> factors) const
{
    validate_model(model, dt, factors);
    auto snapshot = sample();
    LumpedInnerCavityStep result;
    collective(*d_comm, [&] { result = model.preview_samples(dt, snapshot.surfaces, factors); });
    validate_result(result);
    attach(std::move(snapshot), result.exchange);
    return result;
}

LumpedInnerCavityStep InnerCavityFieldInput::advance(
    LumpedInnerCavityModel& model, real_t dt, std::span<const real_t> factors) const
{
    auto result = preview(model, dt, factors);
    std::optional<LumpedInnerCavityModel> candidate;
    collective(*d_comm,
        [&]
        {
            candidate.emplace(model);
            candidate->set_state(result.state);
        });
    model = std::move(*candidate);
    return result;
}

} // namespace SimpleFluid
