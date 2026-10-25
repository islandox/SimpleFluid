/** @file Reduction.cc
 * @brief Owner-routed and budgeted exact inventory sum.
 */
#include "solvers/details/GeneralVoF/exact/Reduction.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include <limits>
#include <set>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof::exact
{
Measure global_sum(const MeshHandle<>& mesh_view, std::span<const Measure> values, std::size_t byte_budget)
{
    const auto* mesh = &mesh_view;
    const Execution execution(mesh_view);
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> local;
    std::vector<OwnerPacket> packets;
    execution.collective(
        [&]
        {
            for (const auto& value : values)
                local = detail::vof_exact::add(local, value, byte_budget);
            packets.push_back({0, local.encode(byte_budget)});
        });
    const auto received = exchange_owner_packets(packets, mesh->owned_cell_map()->getComm(), byte_budget);
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> sum;
    execution.collective(
        [&]
        {
            if (mesh->owned_cell_map()->getComm()->getRank() == 0)
            {
                std::set<int> senders;
                for (const auto& p : received.packets)
                {
                    if (!senders.insert(p.peer).second)
                        throw std::runtime_error("Duplicate exact global inventory sender.");
                    sum = detail::vof_exact::add(sum,
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(
                            p.bytes, byte_budget),
                        byte_budget);
                }
                if (senders.size() != std::size_t(mesh->owned_cell_map()->getComm()->getSize()))
                    throw std::runtime_error("Incomplete exact global inventory reduction.");
            }
        });
    std::vector<char> encoded;
    execution.collective(
        [&]
        {
            if (mesh->owned_cell_map()->getComm()->getRank() == 0)
                encoded = sum.encode(byte_budget);
        });
    std::uint64_t count = encoded.size();
    Teuchos::broadcast(*mesh->owned_cell_map()->getComm(), 0, 1, &count);
    execution.collective(
        [&]
        {
            if (count > byte_budget || count > std::uint64_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("Exact global inventory broadcast exceeds budget.");
            encoded.resize(std::size_t(count));
        });
    Teuchos::broadcast(*mesh->owned_cell_map()->getComm(), 0, int(count), encoded.data());
    execution.collective(
        [&]
        {
            sum = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(encoded, byte_budget);
        });
    return sum;
}
} // namespace SimpleFluid::detail::general_vof::exact
