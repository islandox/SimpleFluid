/**
 * @file BubbleRepresentationInternal.hh
 * @brief Private representation planner owned transactionally by the radiolytic controller.
 */
#pragma once
#include "equations/RadiolyticBubbleKinetics.hh"
#include "solvers/AdaptiveVoFSolver.hh"
#include "solvers/BubbleRepresentation.hh"

#include <map>

namespace SimpleFluid::detail
{
class SIMPLEFLUID_SOLVERS_LOCAL BubbleRepresentationEngine
{
public:
    struct Snapshot
    {
        std::vector<BubbleBirthRecord> births;
        std::vector<BubbleConversionEvent> history;
        std::map<std::uint64_t, std::vector<std::uint64_t>> origins;
        std::uint64_t next_event = 1, source_revision = 0, topology = 0;
    };
    struct Prepared
    {
        BubbleRepresentationInspection inspection;
        VoF::RepresentationChangeRequest request;
        std::vector<std::vector<Meshes::AMRBox>> supports;
        std::shared_ptr<const Snapshot> next;
    };
    BubbleRepresentationEngine(BubbleRepresentationOptions options, RadiolyticGasPhysics::MolarPivotGrid pivots,
        std::vector<std::string> populations);
    void request_births(const std::vector<BubbleBirthSeed>& seeds);
    [[nodiscard]] const Snapshot& snapshot() const noexcept;
    [[nodiscard]] std::shared_ptr<const Snapshot> snapshot_handle() const noexcept;
    [[nodiscard]] std::shared_ptr<const Snapshot> invalidated_snapshot() const;
    void publish(std::shared_ptr<const Snapshot> snapshot, std::uint64_t revision) noexcept;
    [[nodiscard]] Prepared prepare(const AdaptiveVoFSolver& flow, const std::vector<Meshes::AMRBox>& headspace) const;

private:
    BubbleRepresentationOptions options_;
    RadiolyticGasPhysics::MolarPivotGrid pivots_;
    std::vector<std::string> populations_;
    std::shared_ptr<const Snapshot> state_;
    std::uint64_t binding_revision_ = 0;
};
} // namespace SimpleFluid::detail
