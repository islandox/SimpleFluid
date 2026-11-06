/** @file SupportCandidateIndex.hh
 * @brief Conservative candidate ordering for certified represented rectangles.
 */
#pragma once
#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Operation-local index over immutable complete-support inputs. Unsupported
 * shapes, scales or controls retain every original candidate. Visitors and
 * source storage remain borrowed for the duration of this non-reentrant call. */
class SupportCandidateIndex
{
public:
    SupportCandidateIndex(std::span<const SupportOccurrence>, std::size_t, const SupportCharge&);
    [[nodiscard]] bool certified() const noexcept { return certified_; }
    /** Emit original ordinals below limit in ascending order, including closed
     * edge/corner bounds. A changed control stamp resumes exhaustive ordering. */
    void visit(std::size_t source, std::size_t limit, const BorrowedSupportCallback<std::size_t>&);

private:
    static constexpr auto absent = std::numeric_limits<std::size_t>::max();
    struct Bounds
    {
        std::array<long double, 2> lower{}, upper{};
    };
    struct Node
    {
        Bounds bounds;
        std::size_t member = absent, left = absent, right = absent;
    };
    [[nodiscard]] bool stable() const noexcept;
    [[nodiscard]] std::size_t build(std::size_t, std::size_t, unsigned);
    void collect(std::size_t, const Bounds&, std::size_t);
    std::vector<Node> nodes_;
    std::vector<std::size_t> members_;
    std::size_t size_ = 0, root_ = absent;
    int rounding_ = -1;
    std::uint64_t x87_ = 0, simd_ = 0;
    bool certified_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
