/** @file ExactLiveStorage.hh
 * @brief Phase-local bounded immutable exact payload leases for geometry work.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <array>
#include <initializer_list>
#include <map>
#include <vector>
namespace SimpleFluid::VoF::detail
{
/** Phase-local rational payload leases. Immutable identities are deduplicated
 * only while retained handles prove their lifetime. Metadata and a separate
 * arithmetic/pending arena share the caller's unchanged application budget.
 * Leases and tracked slots must not outlive this phase-local owner.
 * Geometry clipping scratch and allocator overhead are outside this contract. */
class SIMPLEFLUID_FVM_EXPORT ExactLiveStorage
{
public:
    using M = Meshes::ExactGeometryMeasure;
    using Budget = Meshes::AMR::adaptive_detail::Budget;
    class Lease
    {
    public:
        Lease() noexcept = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) noexcept;
        Lease& operator=(Lease&&) noexcept;
        ~Lease();

    private:
        friend class ExactLiveStorage;
        Lease(ExactLiveStorage*, const void*) noexcept;
        ExactLiveStorage* owner_ = nullptr;
        const void* identity_ = nullptr;
    };
    class Scope
    {
    public:
        Scope(ExactLiveStorage&, std::initializer_list<M*>);
        Scope(const Scope&) = delete;
        ~Scope();

    private:
        ExactLiveStorage& owner_;
        std::array<M*, 16> values_{};
        std::size_t count_ = 0;
    };
    ExactLiveStorage(Budget&, std::size_t operation_bytes);
    ~ExactLiveStorage();
    ExactLiveStorage(const ExactLiveStorage&) = delete;
    [[nodiscard]] Lease pin(const M&);
    void track(M&);
    void untrack(M&) noexcept;
    void retain(const M&);
    void retain(const Meshes::ExactPhaseMeasures&);
    void replace(M&, const M&);
    void add(M&, const M&);
    void subtract(M&, const M&);
    [[nodiscard]] M sum(const M&, const M&) const;
    [[nodiscard]] M difference(const M&, const M&) const;
    [[nodiscard]] M scaled(const M&, double) const;
    [[nodiscard]] std::size_t operation_bytes() const noexcept { return operation_bytes_; }

private:
    struct Record
    {
        M value;
        std::size_t refs, bytes;
    };
    Budget& budget_;
    std::size_t operation_bytes_, retained_metadata_ = 0;
    std::map<const void*, Record> records_;
    std::map<M*, Lease> slots_;
    std::vector<Lease> retained_;
    void charge(std::size_t);
    void release(const void*) noexcept;
    [[nodiscard]] std::size_t preflight(const M&, const M* = nullptr) const;
    [[nodiscard]] M checked(M, std::size_t) const;
};
} // namespace SimpleFluid::VoF::detail
