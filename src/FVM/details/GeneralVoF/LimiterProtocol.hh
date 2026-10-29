/** @file LimiterProtocol.hh
 * @brief Collective canonical publication of a donor-local limiter replay.
 */
#pragma once
#include "FVM/details/GeneralVoF/LimiterReplay.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <map>
#include <set>
namespace SimpleFluid::VoF::general_vof_detail::limiter_replay
{
// Adapters retain the caller's production wire encoding and subscriber
// distribution. Preparation and final admission run inside collective guards.
// Authority and streams are trial scratch and must be discarded on rejection.
template<class Prepare, class FaceOwner, class WriteReal, class ReadReal, class Distribute, class Admit>
bool synchronize(const Teuchos::RCP<const Teuchos::Comm<int>>& comm, std::size_t limit,
    const std::map<std::uint64_t, GeneralVoFDirectedStream>& streams,
    std::map<std::uint64_t, GeneralVoFDirectedStream>& authority, std::vector<OwnerPacket>& packets,
    const std::array<double, 2>& density, Prepare&& prepare, FaceOwner&& owner, WriteReal&& write_real,
    ReadReal&& read_real, Distribute&& distribute, Admit&& admit)
{
    std::map<std::uint64_t, long double> replacements;
    wire::collective(*comm, [&] { prepare(replacements); });
    int local_repair = replacements.empty() ? 0 : 1, any_repair = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_repair, &any_repair);
    if (!any_repair)
        return false;
    packets.clear();
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [face, s] : streams)
                if (s.donor_owner == comm->getRank())
                {
                    wire::Writer w;
                    w.word(face);
                    w.word(s.donor);
                    write_real(w, replacements.contains(face) ? replacements.at(face) : s.theta);
                    budget.packet(w.data.size());
                    packets.push_back({owner(face), std::move(w.data)});
                }
        });
    const auto proposals = exchange_owner_packets(packets, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<std::uint64_t> seen;
            wire::Budget budget{limit};
            budget.add(authority.size() * (sizeof(std::uint64_t) + 4 * sizeof(void*)));
            for (const auto& p : proposals.packets)
            {
                wire::Reader r{p.bytes};
                const auto face = r.word(), donor = r.word();
                const auto theta = read_real(r);
                r.done();
                if (!authority.contains(face) || !seen.insert(face).second)
                    throw std::invalid_argument("Limiter replacement face is foreign or duplicated.");
                auto& s = authority.at(face);
                if (p.peer != s.donor_owner || donor != s.donor || !std::isfinite(theta) || theta < 0 ||
                    theta > s.theta || (s.theta > 0 && theta == 0))
                    throw std::invalid_argument("Limiter replacement donor or coefficient differs.");
                s.theta = theta;
                evaluate(s, density);
            }
            if (seen.size() != authority.size())
                throw std::invalid_argument("Limiter replacements are incomplete.");
        });
    distribute();
    wire::collective(*comm, [&] { admit(); });
    return true;
}
} // namespace SimpleFluid::VoF::general_vof_detail::limiter_replay
