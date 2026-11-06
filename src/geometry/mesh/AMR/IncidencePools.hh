#pragma once
#include "IdentityPool.hh"
#include "IncidencePointDirectory.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Distinct typed pools prevent accidental equality across semantic domains.
 * Ordinary points are inline; canonical plane owners and shared ancestry use
 * the session identity ledger, separately from pool records and indexes. */
struct IncidencePools
{
    using Point = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
    using Plane = PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>;
    using Image = PLIC::DisplacementIdentity<ArithmeticMode::Ordinary>;
    /** Full precision frame values, with signed zero retained. This is value
     * sharing only: the complete plane support remains its own authority. */
    struct Normal
    {
        std::array<long double, 3> values;
        explicit Normal(const std::array<long double, 3>& source) : values(source)
        {
            for (const auto value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite incidence frame normal.");
        }
        [[nodiscard]] bool operator<(const Normal& other) const noexcept
        {
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (values[i] < other.values[i])
                    return true;
                if (other.values[i] < values[i])
                    return false;
                if (values[i] == 0 && std::signbit(values[i]) != std::signbit(other.values[i]))
                    return std::signbit(values[i]);
            }
            return false;
        }
    };
    IdentityPool<AdaptiveCellKey> cells;
    IdentityPool<AdaptiveBytes> domains, supports, lines;
    IncidencePointDirectory<> points;
    IdentityPool<Plane> planes;
    IdentityPool<Normal> normals;
    IdentityPool<Image> images;
    Budget* budget;
    IdentityBudget* identities;

    explicit IncidencePools(Budget& arena, IdentityBudget& ledger, std::shared_ptr<Budget> keeper = {},
        IdentityPoolStorage storage = IdentityPoolStorage::Flat)
        : cells(arena, keeper, {}, {}, storage), domains(arena, keeper, {}, {}, storage),
          supports(arena, keeper, {}, {}, storage), lines(arena, keeper, {}, {}, storage),
          points(arena, keeper, {}, {}, storage), planes(arena, keeper, {}, {}, storage),
          normals(arena, keeper, {}, {}, storage), images(arena, std::move(keeper), {}, {}, storage), budget(&arena),
          identities(&ledger)
    {
    }
    IncidencePools(const IncidencePools& source, Budget& arena, IdentityBudget& ledger,
        std::shared_ptr<Budget> keeper = {}, std::shared_ptr<CartesianAuthorityArena> source_arena = {})
        : cells(source.cells, arena, keeper), domains(source.domains, arena, keeper),
          supports(source.supports, arena, keeper), lines(source.lines, arena, keeper),
          points(source.points, arena, keeper, std::move(source_arena)), planes(source.planes, arena, keeper),
          normals(source.normals, arena, keeper), images(source.images, arena, std::move(keeper)), budget(&arena),
          identities(&ledger)
    {
        if (!planes.persistent())
            for (std::size_t slot = 0; slot < planes.size(); ++slot)
                identities->retain(arena, planes.get(slot));
    }
    [[nodiscard]] std::size_t intern_plane(const Plane& plane)
    {
        const auto slot = planes.intern(plane);
        if (!planes.persistent())
            identities->retain(*budget, planes.get(slot));
        return slot;
    }
    [[nodiscard]] std::size_t source_only_metadata_bytes() const noexcept
    {
        return cells.source_only_metadata_bytes() + domains.source_only_metadata_bytes() +
               supports.source_only_metadata_bytes() + lines.source_only_metadata_bytes() +
               points.source_only_metadata_bytes() + planes.source_only_metadata_bytes() +
               normals.source_only_metadata_bytes() + images.source_only_metadata_bytes();
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return cells.overlap_bytes() + domains.overlap_bytes() + supports.overlap_bytes() + lines.overlap_bytes() +
               points.overlap_bytes() + planes.overlap_bytes() + normals.overlap_bytes() + images.overlap_bytes();
    }
    void finish_overlap() noexcept
    {
        cells.finish_overlap();
        domains.finish_overlap();
        supports.finish_overlap();
        lines.finish_overlap();
        points.finish_overlap();
        planes.finish_overlap();
        normals.finish_overlap();
        images.finish_overlap();
    }
    [[nodiscard]] std::size_t page_credit_overhead_bytes() const noexcept
    {
        return cells.page_credit_overhead_bytes() + domains.page_credit_overhead_bytes() +
               supports.page_credit_overhead_bytes() + lines.page_credit_overhead_bytes() +
               points.page_credit_overhead_bytes() + planes.page_credit_overhead_bytes() +
               normals.page_credit_overhead_bytes() + images.page_credit_overhead_bytes();
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return cells.metadata_bytes() + domains.metadata_bytes() + supports.metadata_bytes() + lines.metadata_bytes() +
               points.metadata_bytes() + planes.metadata_bytes() + normals.metadata_bytes() + images.metadata_bytes();
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
