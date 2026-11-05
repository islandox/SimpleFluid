#pragma once
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/CartesianIncidenceAuthority.hh"
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
namespace cartesian_incidence_test
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes::AMR;
using namespace SimpleFluid::Meshes::AMR::adaptive_detail;
inline constexpr std::size_t ample = 64 * 1024 * 1024;
inline AMRGrid grid()
{
    AMRGrid value;
    value.block_cells = {2, 2, 2};
    value.max_level = 1;
    return value;
}
inline std::vector<AdaptiveLeaf> leaves(const AMRGrid& value, bool refined_corner = false)
{
    std::vector<AdaptiveLeaf> result;
    for (std::uint64_t z = 0; z != 2; ++z)
        for (std::uint64_t y = 0; y != 2; ++y)
            for (std::uint64_t x = 0; x != 2; ++x)
            {
                if (!refined_corner || x || y || z)
                    result.push_back(
                        cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {x, y, z}}}, {'d'}));
                if (refined_corner)
                    result.push_back(
                        cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{1, {x, y, z}}}, {'d'}));
            }
    return result;
}
inline auto point(long double x, long double y = 0, long double z = 0)
{
    return PLIC::details::GeometryIdentityAccess::point({x, y, z});
}
struct RestoreEnvironment
{
    std::fenv_t saved{};
    RestoreEnvironment()
    {
        if (std::fegetenv(&saved))
            throw std::runtime_error("Cannot save test environment.");
    }
    ~RestoreEnvironment() { (void) std::fesetenv(&saved); }
};
} // namespace cartesian_incidence_test
