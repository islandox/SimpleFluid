#include <utility>
/** @file ConservativeVoFExactArithmetic.cc
 * @brief Guarded exact scalar growth and budgeted collective scalar certificates.
 */
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
namespace SimpleFluid::detail::vof_exact
{
namespace
{
using Map = DefaultTpetraTypes::map_type;
void retained(const Measure& value, std::size_t budget)
{
    const auto bytes = value.storage_bytes();
    if (bytes > budget)
        throw std::runtime_error("Exact VoF scalar retained storage exceeds its budget: stored_bytes=" +
                                 std::to_string(bytes) + ", limit=" + std::to_string(budget) + ".");
}
void operands(const Measure& a, const Measure& b, std::size_t budget)
{
    // A conservative admission bound on exact numerator/denominator payloads;
    // backend allocator/GCD temporaries are not a whole-process RSS promise.
    const auto x = a.storage_bytes(), y = b.storage_bytes();
    if (x > budget / 8 || y > budget / 8 - x)
        throw std::runtime_error("Exact VoF aggregate operands exceed their growth budget: x=" + std::to_string(x) +
                                 ", y=" + std::to_string(y) + ", scalar_budget=" + std::to_string(budget) + ".");
}
template<class Work> void collective(const Map& map, Work&& work)
{
    std::exception_ptr error;
    try
    {
        std::forward<Work>(work)();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = bool(error), global = 0;
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
    if (global)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("Exact VoF scalar reduction rejected on another rank.");
    }
}
void word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.push_back(char(value >> (8 * i)));
}
std::uint64_t word(const std::vector<char>& bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < 8)
        throw std::runtime_error("Truncated exact VoF scalar framing.");
    std::uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i)
        result |= std::uint64_t(static_cast<unsigned char>(bytes[offset + i])) << (8 * i);
    return result;
}
void append(std::vector<char>& bytes, const Measure& value, std::size_t budget)
{
    const auto encoded = value.encode(budget);
    if (bytes.size() > budget || budget - bytes.size() < 8 || encoded.size() > budget - bytes.size() - 8)
        throw std::runtime_error("Exact VoF scalar certificate exceeds its packet budget.");
    word(bytes, encoded.size());
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
}
Measure read(const std::vector<char>& bytes, std::size_t& offset, std::size_t budget)
{
    const auto n = word(bytes, offset);
    offset += 8;
    if (n > bytes.size() - offset)
        throw std::runtime_error("Invalid exact VoF scalar payload length.");
    const auto value = Measure::decode(std::span<const char>(bytes.data() + offset, std::size_t(n)), budget);
    offset += std::size_t(n);
    return value;
}
} // namespace
Measure add(const Measure& a, const Measure& b, std::size_t budget)
{
    if (!a.sign())
    {
        retained(b, budget);
        return b;
    }
    if (!b.sign())
    {
        retained(a, budget);
        return a;
    }
    operands(a, b, budget);
    auto value = a + b;
    retained(value, budget);
    return value;
}
Measure scale(const Measure& a, double scalar, std::size_t budget)
{
    operands(a, Measure::from_value(scalar), budget);
    auto value = a.scaled(scalar);
    retained(value, budget);
    return value;
}
Measure divide(const Measure& a, const Measure& b, std::size_t budget)
{
    return a.divided_by(b, budget);
}
Measure square(const Measure& a, std::size_t budget)
{
    return a.squared(budget);
}
namespace
{
std::size_t operation_lease(const Measure& a, const Measure& b, std::size_t cap)
{
    retained(a, cap);
    retained(b, cap);
    if (cap > std::numeric_limits<std::size_t>::max() / 16)
        throw std::runtime_error("Exact VoF temporary operation lease overflow.");
    return 16 * cap;
}
} // namespace
Measure leased_add(const Measure& a, const Measure& b, std::size_t cap)
{
    const auto result = add(a, b, operation_lease(a, b, cap));
    retained(result, cap);
    return result;
}
Measure leased_scale(const Measure& a, double scalar, std::size_t cap)
{
    const auto result = scale(a, scalar, operation_lease(a, Measure::from_value(scalar), cap));
    retained(result, cap);
    return result;
}
Measure leased_divide(const Measure& a, const Measure& b, std::size_t cap)
{
    const auto result = divide(a, b, operation_lease(a, b, cap));
    retained(result, cap);
    return result;
}
double view(const Measure& value, SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>& enclosure)
{
    enclosure = value.enclosure();
    const double result = static_cast<double>(enclosure.value);
    if (!std::isfinite(result))
        throw std::runtime_error("Exact VoF inner view overflows binary64.");
    auto error = enclosure.absolute_error + std::abs(enclosure.value - static_cast<long double>(result));
    if (error > 0)
        error = std::nextafter(error, std::numeric_limits<long double>::infinity());
    enclosure.value = result;
    enclosure.absolute_error = error;
    return result;
}
Measure maximum_absolute(const std::vector<Measure>& values, const Map& map, std::size_t budget)
{
    std::vector<OwnerPacket> packets;
    collective(map,
        [&]
        {
            Measure maximum;
            for (const auto& value : values)
            {
                const auto magnitude = value.sign() < 0 ? scale(value, -1, budget) : value;
                if (magnitude.compare(maximum) > 0)
                    maximum = magnitude;
            }
            OwnerPacket packet;
            packet.peer = 0;
            append(packet.bytes, maximum, budget);
            packets.push_back(std::move(packet));
        });
    const auto received = exchange_owner_packets(packets, map.getComm(), budget);
    Measure maximum;
    collective(map,
        [&]
        {
            if (map.getComm()->getRank() != 0)
                return;
            std::set<int> senders;
            for (const auto& packet : received.packets)
            {
                if (!senders.insert(packet.peer).second)
                    throw std::runtime_error("Duplicate exact VoF maximum aggregate.");
                std::size_t offset = 0;
                const auto candidate = read(packet.bytes, offset, budget);
                if (offset != packet.bytes.size() || candidate.sign() < 0)
                    throw std::runtime_error("Invalid exact VoF maximum aggregate.");
                if (candidate.compare(maximum) > 0)
                    maximum = candidate;
            }
            if (senders.size() != std::size_t(map.getComm()->getSize()))
                throw std::runtime_error("Incomplete exact VoF maximum reduction.");
        });
    std::vector<char> encoded;
    collective(map,
        [&]
        {
            if (map.getComm()->getRank() == 0)
                encoded = maximum.encode(budget);
        });
    std::uint64_t size = encoded.size();
    Teuchos::broadcast(*map.getComm(), 0, 1, &size);
    collective(map,
        [&]
        {
            if (size > budget || size > std::uint64_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("Exact VoF maximum broadcast exceeds its budget.");
            encoded.resize(std::size_t(size));
        });
    Teuchos::broadcast(*map.getComm(), 0, int(size), encoded.data());
    collective(map, [&] { maximum = Measure::decode(encoded, budget); });
    return maximum;
}
Measure leased_maximum_absolute(const std::vector<Measure>& values, const Map& map, std::size_t cap, std::size_t arena)
{
    const auto comm = map.getComm();
    Measure maximum;
    std::size_t decode_cap = 0;
    collective(map,
        [&]
        {
            if (cap > std::numeric_limits<std::size_t>::max() / 16 || arena < 512)
                throw std::runtime_error("Invalid leased maximum scalar/communication admission.");
            // Valid local certificates fit K plus framing. This larger decoding cap
            // remains inside the separately reserved16*K operation workspace;
            // decoded results must still pass the persistent K check.
            decode_cap = std::min(arena, 2 * cap);
        });
    {
        std::vector<OwnerPacket> packets;
        collective(map,
            [&]
            {
                for (const auto& value : values)
                {
                    retained(value, cap);
                    const auto magnitude = value.sign() < 0 ? leased_scale(value, -1, cap) : value;
                    if (magnitude.compare(maximum) > 0)
                        maximum = magnitude;
                }
                packets.reserve(1);
                OwnerPacket packet;
                packet.peer = 0;
                // Encoding and its packet copy coexist only in this construction
                // phase. The quarter-arena export cap leaves room for both.
                append(packet.bytes, maximum, arena / 4);
                packets.push_back(std::move(packet));
                if (packets.capacity() > (arena / 4) / sizeof(OwnerPacket))
                    throw std::runtime_error("Leased maximum export metadata exceeds communication share.");
                std::size_t resident = packets.capacity() * sizeof(OwnerPacket);
                for (const auto& item : packets)
                {
                    if (item.bytes.capacity() > arena / 4 - resident)
                        throw std::runtime_error("Leased maximum exports exceed their communication share.");
                    resident += item.bytes.capacity();
                }
                if (comm->getRank() == 0 && std::size_t(comm->getSize()) > arena / 4)
                    throw std::runtime_error("Leased maximum sender validation exceeds communication arena.");
            });
        const auto received = exchange_owner_packets(packets, comm, arena / 2);
        collective(map,
            [&]
            {
                if (comm->getRank() != 0)
                    return;
                std::vector<unsigned char> seen(std::size_t(comm->getSize()), 0);
                std::size_t count = 0;
                for (const auto& packet : received.packets)
                {
                    if (packet.peer < 0 || packet.peer >= comm->getSize() || seen[std::size_t(packet.peer)]++)
                        throw std::runtime_error("Duplicate/invalid leased maximum sender.");
                    ++count;
                    std::size_t offset = 0;
                    const auto candidate = read(packet.bytes, offset, decode_cap);
                    retained(candidate, cap);
                    if (offset != packet.bytes.size() || candidate.sign() < 0)
                        throw std::runtime_error("Invalid leased maximum aggregate.");
                    if (candidate.compare(maximum) > 0)
                        maximum = candidate;
                }
                if (count != std::size_t(comm->getSize()))
                    throw std::runtime_error("Incomplete leased maximum reduction.");
            });
    } // Release caller/export/received buffers before broadcast allocation.
    std::vector<char> encoded;
    collective(map,
        [&]
        {
            if (comm->getRank() == 0)
                encoded = maximum.encode(arena / 2);
        });
    std::uint64_t size = encoded.size();
    Teuchos::broadcast(*comm, 0, 1, &size);
    collective(map,
        [&]
        {
            if (size > arena / 2 || size > decode_cap || size > std::uint64_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("Leased maximum broadcast exceeds communication/scalar admission.");
            encoded.resize(std::size_t(size));
            if (encoded.capacity() > arena)
                throw std::runtime_error("Leased maximum broadcast capacity exceeds communication arena.");
        });
    Teuchos::broadcast(*comm, 0, int(size), encoded.data());
    collective(map,
        [&]
        {
            maximum = Measure::decode(encoded, decode_cap);
            retained(maximum, cap);
        });
    return maximum;
}
PhysicalNorm leased_physical_norm(const std::vector<Measure>& residual, const std::vector<Measure>& rhs,
    double tolerance, const Map& map, std::size_t total_budget)
{
    const auto comm = map.getComm();
    const auto communication = total_budget / 4;
    const auto cap = (total_budget - communication) / 24;
    const auto operation = 16 * cap;
    collective(map,
        [&]
        {
            if (cap < 1024 || residual.size() != rhs.size())
                throw std::runtime_error("Leased exact VoF norm has invalid shape or insufficient total arena.");
        });
    // 8K pending +16K operation +C <= B. Compaction only removes allocator
    // slack; exact residuals and the original zero-RHS reference remain intact.
    const auto keep = [&](Measure value)
    {
        if (value.storage_bytes() > cap)
            value = value.compacted(operation);
        retained(value, cap);
        return value;
    };
    const auto sum = [&](const Measure& a, const Measure& b)
    {
        retained(a, cap);
        retained(b, cap);
        return keep(add(a, b, operation));
    };
    const auto squared = [&](const Measure& value)
    {
        retained(value, cap);
        return keep(square(value, operation));
    };
    const auto scaled = [&](const Measure& value, double factor)
    {
        retained(value, cap);
        return keep(scale(value, factor, operation));
    };
    PhysicalNorm result;
    {
        std::vector<OwnerPacket> packets;
        collective(map,
            [&]
            {
                Measure r, b;
                for (std::size_t i = 0; i < rhs.size(); ++i)
                {
                    r = sum(r, squared(residual[i]));
                    b = sum(b, squared(rhs[i]));
                }
                const auto first = r.storage_bytes(), second = b.storage_bytes();
                const auto codec = communication / 8;
                if (first > codec || second > codec - first || codec - first - second < 128)
                    throw std::runtime_error("Leased exact VoF norm codec exceeds communication arena.");
                const auto er = r.encode(first + 56), eb = b.encode(second + 56);
                if (er.capacity() > 2 * (first + 56) || eb.capacity() > 2 * (second + 56))
                    throw std::runtime_error("Leased exact VoF norm encoding capacity exceeds admission.");
                const auto bytes = er.size() + eb.size() + 16;
                const auto exports = communication / 4;
                if (exports < 2 * sizeof(OwnerPacket) || bytes > (exports - 2 * sizeof(OwnerPacket)) / 2)
                    throw std::runtime_error("Leased exact VoF norm exports exceed communication arena.");
                OwnerPacket packet;
                packet.peer = 0;
                packet.bytes.reserve(bytes);
                if (packet.bytes.capacity() > 2 * bytes)
                    throw std::runtime_error("Leased exact VoF norm export capacity exceeds admission.");
                word(packet.bytes, er.size());
                packet.bytes.insert(packet.bytes.end(), er.begin(), er.end());
                word(packet.bytes, eb.size());
                packet.bytes.insert(packet.bytes.end(), eb.begin(), eb.end());
                packets.reserve(1);
                if (packets.capacity() * sizeof(OwnerPacket) + packet.bytes.capacity() > exports)
                    throw std::runtime_error("Leased exact VoF norm packet capacity exceeds arena.");
                packets.push_back(std::move(packet));
            });
        const auto received = exchange_owner_packets(packets, comm, communication / 2);
        collective(map,
            [&]
            {
                if (comm->getRank() != 0)
                    return;
                if (std::size_t(comm->getSize()) > communication / 4)
                    throw std::runtime_error("Leased exact VoF norm sender metadata exceeds arena.");
                std::vector<unsigned char> seen(std::size_t(comm->getSize()));
                if (seen.capacity() > communication / 4)
                    throw std::runtime_error("Leased exact VoF norm sender capacity exceeds arena.");
                Measure r, b;
                std::size_t count = 0;
                for (const auto& packet : received.packets)
                {
                    if (packet.peer < 0 || packet.peer >= comm->getSize() || seen[std::size_t(packet.peer)]++)
                        throw std::runtime_error("Duplicate exact VoF residual aggregate.");
                    ++count;
                    std::size_t offset = 0;
                    const auto first = keep(read(packet.bytes, offset, 2 * cap));
                    r = sum(r, first);
                    const auto second = keep(read(packet.bytes, offset, 2 * cap));
                    b = sum(b, second);
                    if (offset != packet.bytes.size())
                        throw std::runtime_error("Trailing exact VoF residual bytes.");
                }
                if (count != std::size_t(comm->getSize()) || r.sign() < 0 || b.sign() < 0)
                    throw std::runtime_error("Invalid exact VoF physical norm.");
                const auto reference = b.sign() ? b : Measure::from_value(1);
                const auto bound = scaled(scaled(reference, tolerance), tolerance);
                result.accepted = r.compare(bound) <= 0;
                const auto ratio = keep(divide(r, reference, operation));
                try
                {
                    result.relative_residual = static_cast<double>(std::sqrt(ratio.enclosure().value));
                }
                catch (...)
                {
                    result.relative_residual = std::numeric_limits<double>::infinity();
                }
            });
    } // All packets/codec buffers die before the scalar acceptance broadcast.
    int accepted = result.accepted;
    Teuchos::broadcast(*comm, 0, 1, &accepted);
    Teuchos::broadcast(*comm, 0, 1, &result.relative_residual);
    result.accepted = accepted;
    return result;
}
PhysicalNorm physical_norm(const std::vector<Measure>& residual, const std::vector<Measure>& rhs, double tolerance,
    const Map& map, std::size_t budget)
{
    std::vector<OwnerPacket> packets;
    collective(map,
        [&]
        {
            if (residual.size() != rhs.size())
                throw std::runtime_error("Exact VoF original residual dimensions differ.");
            Measure r, b;
            for (std::size_t i = 0; i < rhs.size(); ++i)
            {
                r = add(r, square(residual[i], budget), budget);
                b = add(b, square(rhs[i], budget), budget);
            }
            OwnerPacket packet;
            packet.peer = 0;
            append(packet.bytes, r, budget);
            append(packet.bytes, b, budget);
            packets.push_back(std::move(packet));
        });
    const auto received = exchange_owner_packets(packets, map.getComm(), budget);
    PhysicalNorm result;
    collective(map,
        [&]
        {
            if (map.getComm()->getRank() != 0)
                return;
            Measure r, b;
            std::set<int> senders;
            for (const auto& packet : received.packets)
            {
                if (!senders.insert(packet.peer).second)
                    throw std::runtime_error("Duplicate exact VoF residual aggregate.");
                std::size_t offset = 0;
                r = add(r, read(packet.bytes, offset, budget), budget);
                b = add(b, read(packet.bytes, offset, budget), budget);
                if (offset != packet.bytes.size())
                    throw std::runtime_error("Trailing exact VoF residual bytes.");
            }
            if (senders.size() != std::size_t(map.getComm()->getSize()) || r.sign() < 0 || b.sign() < 0)
                throw std::runtime_error("Invalid exact VoF physical norm.");
            // Preserve the scalar implementation's zero-RHS absolute reference.
            const auto reference = b.sign() ? b : Measure::from_value(1);
            const auto bound = scale(scale(reference, tolerance, budget), tolerance, budget);
            result.accepted = r.compare(bound) <= 0;
            const auto ratio = divide(r, reference, budget);
            try
            {
                result.relative_residual = static_cast<double>(std::sqrt(ratio.enclosure().value));
            }
            catch (...)
            {
                result.relative_residual = std::numeric_limits<double>::infinity();
            }
        });
    int accepted = result.accepted;
    Teuchos::broadcast(*map.getComm(), 0, 1, &accepted);
    Teuchos::broadcast(*map.getComm(), 0, 1, &result.relative_residual);
    result.accepted = accepted;
    return result;
}
} // namespace SimpleFluid::detail::vof_exact
