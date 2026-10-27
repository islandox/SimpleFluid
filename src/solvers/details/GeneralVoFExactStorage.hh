/** @file GeneralVoFExactStorage.hh
 * @brief Cumulative exact payload and explicit container capacity accounting.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
#include <array>
#include <map>
namespace SimpleFluid::detail
{
enum class GeneralVoFStorageDomain : unsigned
{
    Input,
    Scratch,
    Result
};
struct GeneralVoFExactStorageReport
{
    std::array<std::size_t, 3> bytes{};
    std::size_t audit_scratch = 0;
    [[nodiscard]] std::size_t total() const;
};
/** Explicit buffers and retained exact payloads, deduplicated by storage_identity.
 * Shared payloads count once, Input before Scratch before Result. Metadata is
 * charged conservatively before allocations. Backend/Krylov allocator internals
 * are outside this contract. Scalar operations receive remaining() BEFORE work;
 * their own bit-growth preflight still applies. */
class SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFExactStorage
{
public:
    explicit GeneralVoFExactStorage(std::size_t byte_budget) : budget_(byte_budget) {}
    void capacity(GeneralVoFStorageDomain, std::size_t count, std::size_t item_bytes);
    void measure(
        GeneralVoFStorageDomain, const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&);
    void release(
        GeneralVoFStorageDomain, const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>&);
    [[nodiscard]] std::size_t remaining() const noexcept { return budget_ - used_; }
    [[nodiscard]] GeneralVoFExactStorageReport finish() const;

private:
    struct Record
    {
        std::size_t bytes = 0;
        std::array<std::size_t, 3> references{};
    };
    std::size_t budget_, used_ = 0;
    std::array<std::size_t, 3> capacity_{};
    std::map<const void*, Record, std::less<const void*>> records_;
};
} // namespace SimpleFluid::detail
