#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include "geometry/mesh/AMR/SupportCandidateIndex.hh"
#include "geometry/mesh/AMR/CartesianIncidenceBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
class ConstructionScope
{
public:
    ConstructionScope(const SupportConstructionAdmission* admission, SupportConstructionKind kind,
        const SupportFace& source, std::span<const SupportFace> others)
        : admission_(admission)
    {
        if (admission_)
            admission_->begin(kind, source, others);
    }
    ~ConstructionScope()
    {
        if (admission_)
            admission_->end();
    }

private:
    const SupportConstructionAdmission* admission_;
};
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
/** Bound one live operation of the unchanged convex kernel. Partition checks
 * retain their conservative envelope over the actual covering pieces; unrelated
 * members of a complete support do not enlarge that construction lease. */
std::size_t support_construction_capacity_bound(SupportConstructionKind kind, const SupportFace& source,
    std::span<const SupportFace> others)
{
    using namespace cartesian_detail;
    auto vertices = source->points().size(), encoded = source->encoded_incidence_size();
    for (const auto& face : others)
    {
        vertices = std::max(vertices, face->points().size());
        encoded = std::max(encoded, face->encoded_incidence_size());
    }
    const auto polygon = product(2, vertices);
    auto per_pair = product(8, PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(polygon));
    per_pair = add(per_pair, product(add(vertices, 4), growth(polygon, 3 * sizeof(long double))));
    per_pair = add(per_pair, product(16, encoded));
    per_pair = add(per_pair, product(polygon, 256));
    if (kind == SupportConstructionKind::Pair)
        return per_pair;
    const auto members = add(others.size(), 1);
    return product(product(members, members), per_pair);
}
void overlay_complete_support(std::span<const SupportOccurrence> entries, std::size_t budget, const SupportEmit& emit,
    const SupportCharge& charge, const SupportChargeFace& charge_face, const SupportConstructionAdmission* construction)
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
    SupportCandidateIndex candidates(entries, budget, charge);
    std::set<AdaptiveCellKey> unique;
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const auto& a = entries[i];
        if (a.domain.empty() || a.domain != entries[0].domain || a.exterior != exterior || a.boundary != boundary ||
            unique.contains(a.cell) || (!exterior && !boundary.empty()))
            throw std::invalid_argument("Inconsistent support membership.");
        charge(sizeof(AdaptiveCellKey) + 4 * sizeof(void*) + key_capacity(a.cell));
        unique.insert(a.cell);
        auto same_side = [&](std::size_t j)
        {
            if (a.orientation == entries[j].orientation)
            {
                ConstructionScope scope(construction, SupportConstructionKind::Pair, a.face,
                    std::span<const SupportFace>(&entries[j].face, 1));
                if (a.face->intersection(*entries[j].face))
                    throw std::invalid_argument("Same-side support overlap.");
            }
        };
        candidates.visit(i, i, same_side);
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
        {
            auto opposite = [&](std::size_t j)
            {
                if (entries[j].orientation >= 0)
                    return;
                ConstructionScope scope(construction, SupportConstructionKind::Pair, entries[i].face,
                    std::span<const SupportFace>(&entries[j].face, 1));
                auto face = entries[i].face->intersection(*entries[j].face);
                if (!face)
                    return;
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
            };
            candidates.visit(i, entries.size(), opposite);
        }
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        ConstructionScope scope(construction, SupportConstructionKind::Partition, entries[i].face, coverage[i]);
        entries[i].face->certify_partition(coverage[i]);
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
