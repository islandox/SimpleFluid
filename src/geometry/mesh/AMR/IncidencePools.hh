#pragma once
#include "IdentityPool.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

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

    static std::uint64_t cell_hash(const AdaptiveCellKey& key) noexcept
    {
        std::uint64_t hash = 14695981039346656037ULL;
        const auto word = [&](std::uint64_t value)
        {
            for (unsigned i = 0; i < 8; ++i)
            {
                hash ^= static_cast<unsigned char>(value >> (8 * i));
                hash *= 1099511628211ULL;
            }
        };
        word(key.region);
        word(key.ancestry.index());
        std::visit(
            [&](const auto& ancestry)
            {
                using A = std::decay_t<decltype(ancestry)>;
                if constexpr (std::is_same_v<A, CartesianCellAncestry>)
                {
                    word(ancestry.level);
                    for (const auto value : ancestry.index)
                        word(value);
                }
                else if constexpr (std::is_same_v<A, ConvexCellAncestry>)
                {
                    word(ancestry.root);
                    word(ancestry.path.size());
                    for (const auto value : ancestry.path)
                        word(value);
                }
                else if constexpr (std::is_same_v<A, ProductCellAncestry>)
                {
                    word(ancestry.base.root);
                    word(ancestry.base.root_template);
                    word(ancestry.base.path.size());
                    for (const auto value : ancestry.base.path)
                        word(value);
                    word(ancestry.axial.root);
                    word(ancestry.axial.path.size());
                    for (const auto value : ancestry.axial.path)
                        word(value);
                }
                else
                {
                    word(ancestry.provider_root);
                    word(ancestry.native_cell);
                }
            },
            key.ancestry);
        return hash;
    }
    static std::uint64_t bytes_hash(const AdaptiveBytes& bytes) noexcept { return adaptive_route_hash(bytes); }
    static std::uint64_t point_hash(const Point& point)
    {
        // Canonical scalar emission uses floating-to-integer conversion. Keep
        // this lookup-only work from changing the caller's flags or trap mask.
        std::fenv_t saved;
        if (std::feholdexcept(&saved) != 0)
            throw std::runtime_error("Cannot isolate point lookup floating-point environment.");
        const auto hash = PLIC::canonical_key_hash(point);
        if (std::fesetenv(&saved) != 0)
            throw std::runtime_error("Cannot restore point lookup floating-point environment.");
        return hash;
    }
    explicit IncidencePools(Budget& arena, IdentityBudget& ledger)
        : cells(
              arena, [](const auto& value) { return key_capacity(value); }, &cell_hash),
          domains(
              arena, [](const auto& value) { return value.capacity(); }, &bytes_hash),
          supports(
              arena, [](const auto& value) { return value.capacity(); }, &bytes_hash),
          lines(
              arena, [](const auto& value) { return value.capacity(); }, &bytes_hash),
          points(
              arena, [](const auto& value) { return value.payload_bytes(); }, &point_hash),
          planes(arena, [](const auto&) -> std::size_t { return 0; }),
          normals(arena, [](const auto&) -> std::size_t { return 0; }),
          images(arena, [](const auto& value) { return value.payload_bytes(); }), budget(&arena), identities(&ledger)
    {
    }
    IncidencePools(const IncidencePools& source, Budget& arena, IdentityBudget& ledger)
        : cells(source.cells, arena), domains(source.domains, arena), supports(source.supports, arena),
          lines(source.lines, arena), points(source.points, arena), planes(source.planes, arena),
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
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return cells.metadata_bytes() + domains.metadata_bytes() + supports.metadata_bytes() + lines.metadata_bytes() +
               points.metadata_bytes() + planes.metadata_bytes() + normals.metadata_bytes() + images.metadata_bytes();
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
