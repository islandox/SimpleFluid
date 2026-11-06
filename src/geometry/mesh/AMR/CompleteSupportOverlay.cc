#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
void append_coverage(std::vector<SupportFace>& row, const SupportFace& face, std::size_t budget,
    const SupportCharge& charge)
{
    if (row.size() == row.capacity())
    {
        const auto previous = row.capacity();
        if (previous > budget / sizeof(SupportFace) / 2)
            throw std::invalid_argument("Authority coverage capacity budget exhausted.");
        const auto next = previous ? 2 * previous : 1;
        if (next > row.max_size())
            throw std::invalid_argument("Authority coverage capacity exceeds container limits.");
        // Admit the new allocation while its predecessor is still live. The
        // caller's scoped charge interface conservatively retains growth
        // history until the complete support operation finishes.
        charge(next * sizeof(SupportFace));
        row.reserve(next);
        if (row.capacity() != next)
            throw std::invalid_argument("Authority coverage capacity differs from admission.");
    }
    row.push_back(face);
}
} // namespace
void overlay_complete_support(std::span<const SupportOccurrence> entries, std::size_t budget, const SupportEmit& emit,
    const SupportCharge& charge, const SupportChargeFace& charge_face)
{
    if (entries.empty())
        throw std::invalid_argument("Empty complete support.");
    const bool exterior = entries[0].exterior;
    const auto& boundary = entries[0].boundary;
    // Preserve the previous conservative pair-count ceiling until certified
    // candidate pruning replaces the all-pairs loops. This is a work guard,
    // not a reservation for nonexistent coverage pointers.
    if (entries.size() > budget / entries.size() / sizeof(SupportFace))
        throw std::invalid_argument("Authority pair-work budget exhausted.");
    std::set<AdaptiveCellKey> unique;
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const auto& a = entries[i];
        if (a.domain.empty() || a.domain != entries[0].domain || a.exterior != exterior || a.boundary != boundary ||
            unique.contains(a.cell) || (!exterior && !boundary.empty()))
            throw std::invalid_argument("Inconsistent support membership.");
        charge(sizeof(AdaptiveCellKey) + 4 * sizeof(void*) + key_capacity(a.cell));
        unique.insert(a.cell);
        for (std::size_t j = 0; j < i; ++j)
            if (a.orientation == entries[j].orientation && a.face->intersection(*entries[j].face))
                throw std::invalid_argument("Same-side support overlap.");
    }
    if (exterior)
    {
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].orientation != entries[0].orientation)
                throw std::invalid_argument("Exterior support has opposite side.");
            emit(i, {}, entries[i].face);
        }
        return;
    }
    if (entries.size() > budget / sizeof(std::vector<SupportFace>))
        throw std::invalid_argument("Authority coverage owner budget exhausted.");
    charge(entries.size() * sizeof(std::vector<SupportFace>));
    std::vector<std::vector<SupportFace>> coverage(entries.size());
    charge((coverage.capacity() - coverage.size()) * sizeof(decltype(coverage)::value_type));
    for (std::size_t i = 0; i < entries.size(); ++i)
        if (entries[i].orientation > 0)
            for (std::size_t j = 0; j < entries.size(); ++j)
                if (entries[j].orientation < 0)
                {
                    auto face = entries[i].face->intersection(*entries[j].face);
                    if (!face)
                        continue;
                    if (entries[i].native_whole && entries[j].native_whole &&
                        !face->equivalent(*entries[i].whole_geometry) && !face->equivalent(*entries[j].whole_geometry))
                        throw std::invalid_argument(
                            "Crossed procedural traces need compatible native closure; mortar tiles are not admitted.");
                    if (!entries[i].face->oppositely_oriented(*entries[j].face))
                        throw std::invalid_argument("Interior support normals disagree.");
                    charge_face(face);
                    append_coverage(coverage[i], face, budget, charge);
                    append_coverage(coverage[j], face, budget, charge);
                    emit(i, j, std::move(face));
                }
    for (std::size_t i = 0; i < entries.size(); ++i)
        entries[i].face->certify_partition(coverage[i]);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
