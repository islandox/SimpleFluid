#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
void overlay_complete_support(std::span<const SupportOccurrence> entries, std::size_t budget, const SupportEmit& emit,
    const SupportCharge& charge, const SupportChargeFace& charge_face)
{
    if (entries.empty())
        throw std::invalid_argument("Empty complete support.");
    const bool exterior = entries[0].exterior;
    const auto& boundary = entries[0].boundary;
    if (entries.size() > budget / sizeof(std::vector<SupportFace>))
        throw std::invalid_argument("Authority coverage owner budget exhausted.");
    charge(entries.size() * sizeof(std::vector<SupportFace>));
    if (entries.size() > budget / entries.size() / sizeof(SupportFace))
        throw std::invalid_argument("Authority overlap budget exhausted.");
    charge(entries.size() * entries.size() * sizeof(SupportFace));
    std::vector<std::vector<SupportFace>> coverage(entries.size());
    charge((coverage.capacity() - coverage.size()) * sizeof(decltype(coverage)::value_type));
    for (auto& row : coverage)
    {
        row.reserve(entries.size());
        charge((row.capacity() - entries.size()) * sizeof(SupportFace));
    }
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
                    coverage[i].push_back(face);
                    coverage[j].push_back(face);
                    emit(i, j, std::move(face));
                }
    for (std::size_t i = 0; i < entries.size(); ++i)
        entries[i].face->certify_partition(coverage[i]);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
