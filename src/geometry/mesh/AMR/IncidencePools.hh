#pragma once
#include "IdentityPool.hh"
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
    IdentityPool<Point> points;
    IdentityPool<Plane> planes;
    IdentityPool<Normal> normals;
    IdentityPool<Image> images;
    Budget* budget;
    IdentityBudget* identities;

    explicit IncidencePools(Budget& arena, IdentityBudget& ledger, std::shared_ptr<Budget> keeper = {})
        : cells(arena), domains(arena), supports(arena), lines(arena), points(arena, std::move(keeper)), planes(arena),
          normals(arena), images(arena), budget(&arena), identities(&ledger)
    {
    }
    IncidencePools(
        const IncidencePools& source, Budget& arena, IdentityBudget& ledger, std::shared_ptr<Budget> keeper = {})
        : cells(source.cells, arena), domains(source.domains, arena), supports(source.supports, arena),
          lines(source.lines, arena), points(source.points, arena, std::move(keeper)), planes(source.planes, arena),
          normals(source.normals, arena), images(source.images, arena), budget(&arena), identities(&ledger)
    {
        for (std::size_t slot = 0; slot < planes.size(); ++slot)
            identities->retain(arena, planes.get(slot));
    }
    [[nodiscard]] std::size_t intern_plane(const Plane& plane)
    {
        const auto slot = planes.intern(plane);
        identities->retain(*budget, planes.get(slot));
        return slot;
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
