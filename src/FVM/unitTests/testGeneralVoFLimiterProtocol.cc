/** Synthetic protocol integration: captured donor arithmetic, owner exchange and recovery. */
#include "FVM/details/GeneralVoF/LimiterProtocol.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <bit>
#include <gtest/gtest.h>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
namespace replay = SimpleFluid::VoF::general_vof_detail::limiter_replay;
namespace wire = SimpleFluid::VoF::general_vof_detail::wire;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Stream = GeneralVoFDirectedStream;
constexpr std::array<double, 2> density{1, 1000};
void put(wire::Writer& w, long double value)
{
    int exponent = 0;
    const auto mantissa = std::frexp(value, &exponent);
    const double high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
    w.word(std::bit_cast<std::uint64_t>(high));
    w.word(std::bit_cast<std::uint64_t>(low));
    w.word(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
}
long double get(wire::Reader& r)
{
    const auto high = std::bit_cast<double>(r.word()), low = std::bit_cast<double>(r.word());
    const auto exponent = std::bit_cast<std::int64_t>(r.word());
    return std::scalbn(static_cast<long double>(high) + low, int(exponent));
}
std::vector<Stream> captured(int donor)
{
    Stream a, b, z;
    a.face = 65;
    a.volume = 0x8959de30fdeed376p-83L;
    a.gas_low = 0x86ccac7872d913dep-109L;
    a.gas_high = 0xd82bbd41671cd438p-105L;
    a.liquid_low = 0x8959de0f4ac3b559p-83L;
    a.liquid_high = 0x8959dad04ef9cddap-83L;
    b.face = 159;
    b.volume = 0x8ed2cf1791916875p-75L;
    b.gas_low = 0x8c2b96fa8d926af1p-101L;
    b.gas_high = 0x94d31023fcc3ca40p-96L;
    b.liquid_low = 0x8ed2cef486aba9d2p-75L;
    b.liquid_high = 0x8ed2ca70f910488fp-75L;
    a.theta = b.theta = 0xd63f4ff2a1329e21p-64L;
    z.face = 199;
    z.volume = z.liquid_low = z.liquid_high = 0x1p-20L;
    for (auto* s : {&a, &b, &z})
    {
        s->donor = 46;
        s->donor_owner = donor;
        s->recipient_owner = 0;
        replay::evaluate(*s, density);
    }
    return {a, b, z};
}
TEST(GeneralVoFLimiterProtocol, FrozenDonorCrossOwnerCollectiveRejectAndRecover)
{
    if (std::numeric_limits<long double>::digits != 64)
        GTEST_SKIP() << "Frozen replay requires the captured x87 representation.";
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 1 && comm->getSize() != 2 && comm->getSize() != 4)
        GTEST_SKIP() << "Requires one, two or four ranks.";
    const int rank = comm->getRank(), donor = comm->getSize() == 1 ? 0 : 1;
    EXPECT_EQ(donor != 0, comm->getSize() > 1);
    constexpr std::size_t limit = 1024 * 1024;
    const auto initial = captured(donor);
    const std::array<long double, 2> old{0x1.f67d1ac23a12ep-34L, 0x1.ffffff8260b95p-8L};
    SimpleFluid::detail::CompensatedSum<long double> gl, ll, go, lo;
    std::array<long double, 4> demand{}, ratio{1, 1, 1, 1};
    for (const auto& s : initial)
    {
        gl += s.gas_low;
        ll += s.liquid_low;
        go += s.gas;
        lo += s.liquid;
        const auto c = replay::correction(s);
        if (c >= 0)
            demand[2] += c;
        else
            demand[3] -= c;
    }
    const std::array<long double, 4> available{0, 0, old[0] - gl.value(), old[1] - ll.value()};
    for (unsigned p = 2; p < 4; ++p)
        if (demand[p] > 0)
        {
            ratio[p] = std::min(1.L, available[p] / demand[p]);
            if (ratio[p] > 0 && ratio[p] < 1)
                ratio[p] = std::nextafter(ratio[p], 0.L);
        }
    ASSERT_GT(go.value(), old[0]);
    ASSERT_LE(lo.value(), old[1]);
    ASSERT_EQ(ratio[2], initial[0].theta);
    std::map<std::uint64_t, Stream> authority, streams;
    if (rank == 0)
        for (const auto& s : initial)
            authority.emplace(s.face, s);
    if (rank <= 1)
        for (const auto& s : initial)
            streams.emplace(s.face, s);
    replay::Evidence evidence;
    std::vector<OwnerPacket> proposal_packets;
    bool distributed = false, admitted = false;
    // Mode 1 corrupts donor state before exchange; mode 2 sends an invalid
    // coefficient and exercises canonical-owner authentication.
    const auto call = [&](int mode)
    {
        return replay::synchronize(
            comm, limit, streams, authority, proposal_packets, density,
            [&](auto& replacements)
            {
                if (rank != donor)
                    return;
                auto outgoing = initial;
                if (mode == 1)
                    outgoing[0].gas += 0x1p-40L;
                const auto repaired = replay::repair(outgoing, old, available, demand, ratio, density, &evidence);
                if (!repaired)
                    throw std::invalid_argument("Synthetic donor replay rejected.");
                for (const auto& s : *repaired)
                    replacements.emplace(s.face, s.theta);
            },
            [](auto) { return 0; },
            [&](wire::Writer& w, long double theta) { put(w, mode == 2 ? std::nextafter(1.L, 2.L) : theta); }, get,
            [&]
            {
                distributed = true;
                std::vector<OwnerPacket> packets;
                wire::collective(*comm,
                    [&]
                    {
                        if (rank == 0)
                            for (const auto& [face, s] : authority)
                                for (int peer = 0; peer <= donor; ++peer)
                                {
                                    wire::Writer w;
                                    w.word(face);
                                    put(w, s.theta);
                                    packets.push_back({peer, std::move(w.data)});
                                }
                    });
                const auto result = exchange_owner_packets(packets, comm, limit);
                wire::collective(*comm,
                    [&]
                    {
                        for (const auto& p : result.packets)
                        {
                            wire::Reader r{p.bytes};
                            const auto face = r.word();
                            const auto theta = get(r);
                            r.done();
                            if (p.peer != 0 || !streams.contains(face))
                                throw std::invalid_argument("Bad fixture publication.");
                            auto& s = streams.at(face);
                            s.theta = theta;
                            replay::evaluate(s, density);
                        }
                    });
            },
            [&]
            {
                admitted = true;
                if (rank == donor)
                {
                    SimpleFluid::detail::CompensatedSum<long double> gas, liquid;
                    for (const auto& [face, s] : streams)
                    {
                        gas += s.gas;
                        liquid += s.liquid;
                    }
                    if (gas.value() > old[0] || liquid.value() > old[1])
                        throw std::invalid_argument("Synthetic donor export inadmissible.");
                }
            });
    };
    EXPECT_THROW(call(1), std::invalid_argument);
    if (rank <= 1)
        for (const auto& s : initial)
        {
            EXPECT_EQ(streams.at(s.face).theta, s.theta);
            EXPECT_EQ(streams.at(s.face).gas, s.gas);
            EXPECT_EQ(streams.at(s.face).liquid, s.liquid);
            EXPECT_EQ(streams.at(s.face).mass, s.mass);
            if (rank == 0)
            {
                EXPECT_EQ(authority.at(s.face).theta, s.theta);
                EXPECT_EQ(authority.at(s.face).gas, s.gas);
                EXPECT_EQ(authority.at(s.face).liquid, s.liquid);
                EXPECT_EQ(authority.at(s.face).mass, s.mass);
            }
        }
    EXPECT_THROW(call(2), std::invalid_argument);
    EXPECT_FALSE(distributed);
    EXPECT_FALSE(admitted);
    // A rejected trial discards its scratch ledger; construct fresh trial
    // state before retrying, just as the production transaction does.
    authority.clear();
    streams.clear();
    if (rank == 0)
        for (const auto& s : initial)
            authority.emplace(s.face, s);
    if (rank <= 1)
        for (const auto& s : initial)
            streams.emplace(s.face, s);
    EXPECT_TRUE(call(0));
    if (rank == donor)
    {
        EXPECT_TRUE(evidence.accepted);
        EXPECT_GT(evidence.evaluations, 0u);
        EXPECT_LE(evidence.movement[0], evidence.movement_limit[0]);
        EXPECT_LE(evidence.movement[1], evidence.movement_limit[1]);
        EXPECT_LE(evidence.final_export[0], old[0]);
        EXPECT_LE(evidence.final_export[1], old[1]);
    }
    if (rank <= 1)
    {
        EXPECT_LT(streams.at(65).theta, initial[0].theta);
        EXPECT_GT(streams.at(65).gas, 0);
        EXPECT_EQ(streams.at(199).theta, 1);
        EXPECT_EQ(streams.at(199).gas, 0);
        EXPECT_EQ(streams.at(199).liquid, initial[2].liquid);
        EXPECT_EQ(streams.at(199).mass, initial[2].mass);
        for (const auto& [face, s] : streams)
        {
            EXPECT_EQ(s.mass, s.gas + 1000 * s.liquid);
            if (rank == 0)
            {
                EXPECT_EQ(s.theta, authority.at(face).theta);
                EXPECT_EQ(s.gas, authority.at(face).gas);
                EXPECT_EQ(s.liquid, authority.at(face).liquid);
                EXPECT_EQ(s.mass, authority.at(face).mass);
            }
        }
    }
    else
    {
        EXPECT_TRUE(streams.empty());
        EXPECT_TRUE(authority.empty());
    }
    // Compare the actual canonical bytes as values on the remote subscriber;
    // empty ranks still enter every collective.
    for (const auto face : {65u, 159u, 199u})
    {
        wire::Writer encoded;
        if (rank == 0)
        {
            const auto& s = authority.at(face);
            for (const auto value : {s.theta, s.gas, s.liquid, s.mass})
                put(encoded, value);
        }
        encoded.data.resize(96);
        Teuchos::broadcast(*comm, 0, int(encoded.data.size()), encoded.data.data());
        if (rank == donor)
        {
            wire::Reader r{encoded.data};
            std::array<long double, 4> canonical{};
            for (auto& value : canonical)
                value = get(r);
            r.done();
            const auto& s = streams.at(face);
            EXPECT_EQ(s.theta, canonical[0]);
            EXPECT_EQ(s.gas, canonical[1]);
            EXPECT_EQ(s.liquid, canonical[2]);
            EXPECT_EQ(s.mass, canonical[3]);
        }
    }
}
} // namespace
