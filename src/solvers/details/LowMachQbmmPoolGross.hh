/** @file LowMachQbmmPoolGross.hh
 * @brief Corrector packets and interval totals for authorized pool transfers.
 */
#pragma once

#include "solvers/details/LowMachQbmmState.hh"
#include "equations/CollectiveValidation.hh"
#include <Teuchos_CommHelpers.hpp>
#include <limits>

namespace SimpleFluid::Reacting::qbmm_detail
{
struct PoolGross
{
    std::vector<std::vector<double>> rows;
    std::vector<Flow::GrossChannelMetadata> metadata;
    Flow::QbmmPoolDiagnostics totals;
};

SIMPLEFLUID_SOLVERS_LOCAL inline PoolGross pool_gross(const QbmmBulkGasPoolBinding& pool, const Flow::Mesh& mesh,
    const Flow::QbmmSourceCells& cells, const PressureConditionedThermalBatch* thermal,
    std::span<const PressureConditionedThermalEvaluation> evaluations)
{
    PoolGross result;
    const auto& catalog = pool.catalog();
    const size_t ns = catalog.species().size(), width = Flow::qbmm_pool_gross_width(ns), stride = width / 2;
    std::vector<double> local, global;
    std::array<double, 2> local_energy{}, global_energy{};
    std::uint64_t evaluations_count = 0;
    collective_detail::collective_local_validation(mesh, "QBMM directional pool packets",
        [&]
        {
            require(width <= size_t(std::numeric_limits<int>::max()), "QBMM pool channel count exceeds MPI limits.");
            for (size_t d = 0; d < 2; ++d)
                result.metadata.insert(result.metadata.end(), stride,
                    {Flow::GrossResidualStage::QbmmPoolConversion, d, d == 0 ? 1 : -1});
            result.rows.resize(cells.size());
            std::vector<SourceReceiptSum> total(width);
            std::array<SourceReceiptSum, 2> signed_energy;
            for (size_t c = 0; c < cells.size(); ++c)
            {
                std::vector<SourceReceiptSum> row(width);
                bool transferred = false;
                for (const auto& family : cells[c])
                {
                    if (!family)
                        continue;
                    evaluations_count += family->pool_conversion_law_evaluated;
                    for (const auto& event : family->events)
                        for (size_t i = 0; i < event.transfers.size(); ++i)
                        {
                            const auto& allocation = event.transfers[i];
                            const auto* exchange = std::get_if<QbmmPhaseExchange>(&allocation.transfer.counterpart);
                            if (!exchange || !exchange->resolved_pool)
                                continue;
                            transferred = true;
                            const size_t offset = allocation.into_family ? stride : 0;
                            const auto& atom = allocation.into_family ? event.products.at(i) : event.donors.at(0).atom;
                            row[offset] += atom.count;
                            for (size_t s = 0; s < ns; ++s)
                            {
                                const double amount = allocation.transfer.moles[s];
                                row[offset + 1 + s] += amount;
                                row[offset + ns + 1] += static_cast<long double>(amount) * catalog.species()[s].molarMass;
                            }
                            const long double H2 = static_cast<long double>(atom.count) * atom.mass_over_scale *
                                                   atom.mass_over_scale;
                            row[offset + ns + 2] += H2;
                            row[offset + ns + 3] += H2 * atom.mass_over_scale;
                        }
                }
                require(!transferred || (thermal && thermal->cells[c].material_tail.paired_stage),
                    "QBMM conversion requires paired enthalpy receipts.");
                if (thermal)
                    for (const auto& event : evaluations[c].material_events)
                    {
                        if (thermal->cells[c].material_tail.paired_stage != event.stage)
                            continue;
                        const auto& slot = thermal->configuration->stages.at(event.stage).material.at(event.slot);
                        if (slot.donor != pool.phase_index() && slot.recipient != pool.phase_index())
                            continue;
                        const size_t direction = slot.donor == pool.phase_index() ? 1 : 0;
                        require(std::isfinite(event.carried_energy), "QBMM pool carried enthalpy must be finite.");
                        row[direction * stride + ns + 4 + (event.carried_energy < 0)] +=
                            std::abs(event.carried_energy);
                        signed_energy[direction] += event.carried_energy;
                    }
                auto& values = result.rows[c];
                values.resize(width);
                for (size_t i = 0; i < width; ++i)
                {
                    const auto raw = row[i].value();
                    values[i] = static_cast<double>(raw);
                    validate_raw_value(values[i], raw, "QBMM pool gross packet is not representable.");
                    require(values[i] >= 0, "QBMM pool gross packet must be nonnegative.");
                    total[i].append(row[i]);
                }
            }
            local.resize(width);
            global.resize(width);
            for (size_t i = 0; i < width; ++i)
            {
                local[i] = static_cast<double>(total[i].value());
                validate_raw_value(local[i], total[i].value(), "QBMM pool local diagnostic is not representable.");
            }
            for (auto& direction : result.totals.directions)
                direction.species_moles.resize(ns);
            for (size_t d = 0; d < signed_energy.size(); ++d)
            {
                const auto raw = signed_energy[d].value();
                local_energy[d] = static_cast<double>(raw);
                require(std::isfinite(raw) && std::isfinite(local_energy[d]),
                    "QBMM signed local pool enthalpy is not representable.");
            }
        });
    const auto comm = mesh.owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(width), local.data(), global.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &evaluations_count, &result.totals.law_evaluations);
    require(std::ranges::all_of(global, [](double x) { return std::isfinite(x) && x >= 0; }),
        "QBMM global pool diagnostics are not representable.");
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local_energy.size()), local_energy.data(), global_energy.data());
    collective_detail::collective_local_validation(mesh, "QBMM global pool totals",
        [&]
        {
            for (size_t d = 0; d < 2; ++d)
            {
                const size_t offset = d * stride;
                auto& out = result.totals.directions[d];
                out.count = global[offset];
                std::copy_n(global.begin() + offset + 1, ns, out.species_moles.begin());
                out.mass_kg = global[offset + ns + 1];
                out.H2 = global[offset + ns + 2];
                out.H3 = global[offset + ns + 3];
                out.carried_enthalpy_j = global_energy[d];
                require(std::isfinite(out.carried_enthalpy_j), "QBMM signed pool enthalpy is not representable.");
            }
        });
    return result;
}
} // namespace SimpleFluid::Reacting::qbmm_detail
