/** @file DistributedAdaptiveWire.hh
 * @brief Internal bounded owner-message primitives, independent of MPI layout.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <Teuchos_CommHelpers.hpp>
#include <exception>
#include <stdexcept>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget
{
    std::size_t limit, used = 0;
    void add(std::size_t bytes)
    {
        if (bytes > limit - used)
            throw std::invalid_argument("Distributed adaptive payload budget exhausted.");
        used += bytes;
    }
    void packet(std::size_t payload)
    {
        add(sizeof(OwnerPacket));
        add(payload);
    }
};
struct Writer
{
    AdaptiveBytes data;
    void word(std::uint64_t n)
    {
        for (unsigned i = 0; i < 8; ++i)
            data.push_back(char(n >> (8 * i)));
    }
    void bytes(std::span<const char> b)
    {
        word(b.size());
        data.insert(data.end(), b.begin(), b.end());
    }
};
struct Reader
{
    std::span<const char> data;
    std::size_t offset = 0;
    std::uint64_t word()
    {
        if (data.size() - offset < 8)
            throw std::invalid_argument("Truncated distributed adaptive message.");
        std::uint64_t n = 0;
        for (unsigned i = 0; i < 8; ++i)
            n |= std::uint64_t(static_cast<unsigned char>(data[offset++])) << (8 * i);
        return n;
    }
    AdaptiveBytes bytes()
    {
        auto n = word();
        if (n > data.size() - offset)
            throw std::invalid_argument("Invalid distributed adaptive message length.");
        AdaptiveBytes b(data.begin() + offset, data.begin() + offset + n);
        offset += n;
        return b;
    }
    void done()
    {
        if (offset != data.size())
            throw std::invalid_argument("Trailing distributed adaptive bytes.");
    }
};
template<class F> void collective(const Teuchos::Comm<int>& comm, F&& action)
{
    std::exception_ptr error;
    try
    {
        action();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int bad = bool(error), any = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &bad, &any);
    if (any)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("Distributed adaptation rejected on another rank.");
    }
}
inline int authority(std::span<const char> key, int ranks)
{
    return int(adaptive_route_hash(key) % std::uint64_t(ranks));
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
