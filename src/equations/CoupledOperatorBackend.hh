#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace SimpleFluid
{
/** Representation of the true coupled operator; assembled remains the reference. */
enum class CoupledOperatorBackend : std::uint8_t
{
    Assembled,
    BlockComposite
};

inline std::string_view to_string(CoupledOperatorBackend backend)
{
    switch (backend)
    {
        case CoupledOperatorBackend::Assembled:
            return "assembled";
        case CoupledOperatorBackend::BlockComposite:
            return "block_composite";
    }
    throw std::invalid_argument("Invalid coupled operator backend.");
}

inline CoupledOperatorBackend coupled_operator_backend_from_string(std::string_view value)
{
    if (value == "assembled")
        return CoupledOperatorBackend::Assembled;
    if (value == "block_composite")
        return CoupledOperatorBackend::BlockComposite;
    throw std::invalid_argument("Unknown coupled operator backend '" + std::string(value) + "'.");
}
/** Temporary sparse-product lifetime, independent of the true operator backend. */
enum class CoupledWorkspacePolicy : std::uint8_t
{
    CachedProducts,
    StreamedProducts
};

inline std::string_view to_string(CoupledWorkspacePolicy policy)
{
    switch (policy)
    {
        case CoupledWorkspacePolicy::CachedProducts:
            return "cached_products";
        case CoupledWorkspacePolicy::StreamedProducts:
            return "streamed_products";
    }
    throw std::invalid_argument("Invalid coupled workspace policy.");
}

inline CoupledWorkspacePolicy coupled_workspace_policy_from_string(std::string_view value)
{
    if (value == "cached_products")
        return CoupledWorkspacePolicy::CachedProducts;
    if (value == "streamed_products")
        return CoupledWorkspacePolicy::StreamedProducts;
    throw std::invalid_argument("Unknown coupled workspace policy '" + std::string(value) + "'.");
}
} // namespace SimpleFluid
