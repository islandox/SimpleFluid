/** @file GeneralVoFPhiClosureSupport.hh
 * @brief Private packet framing and collective arena admission for Phi forests.
 */
#pragma once
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <Teuchos_CommHelpers.hpp>
#include <exception>
#include <stdexcept>
namespace SimpleFluid::detail::phi_support
{
using GO = DefaultTpetraTypes::global_ordinal_type;
using Bytes = std::vector<char>;
class PhiProposalResource : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
struct SIMPLEFLUID_SOLVERS_LOCAL Codec
{
    Bytes bytes;
    std::size_t cursor = 0, limit = 0, decode_limit = 0, decode_used = 0;
    explicit Codec(std::size_t budget);
    Codec(const Bytes& input, std::size_t budget);
    void put(std::uint64_t value);
    std::uint64_t get();
    void put(const Bytes& value);
    Bytes blob();
    void done();
};
SIMPLEFLUID_SOLVERS_LOCAL std::size_t packet_capacity(const std::vector<OwnerPacket>& packets, std::size_t budget);
SIMPLEFLUID_SOLVERS_LOCAL std::size_t phi_decode_budget(const std::vector<OwnerPacket>& inputs,
    const std::vector<OwnerPacket>& outputs, const Teuchos::Comm<int>& comm, std::size_t arena);
SIMPLEFLUID_SOLVERS_LOCAL OwnerExchange exchange_phi_packets(const std::vector<OwnerPacket>& packets,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm, std::size_t input_cap, std::size_t exchange_cap,
    std::size_t arena_cap, std::size_t* resident_input = nullptr);
} // namespace SimpleFluid::detail::phi_support
#include "solvers/details/GeneralVoFPhiClosureSupport.ipp"
