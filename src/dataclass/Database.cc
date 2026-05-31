/**
 * @file Database.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line non-template method implementations for Database.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "Database.hh"

namespace SimpleFluid
{

bool Database::erase_from_node(const std::string& key, NodeKind kind)
{
    switch (kind)
    {
        case NodeKind::Int:
            return int_node.erase(key);
        case NodeKind::Real:
            return real_node.erase(key);
        case NodeKind::String:
            return string_node.erase(key);
        case NodeKind::Bool:
            return bool_node.erase(key);
        case NodeKind::VecInt:
            return vec_int_node.erase(key);
        case NodeKind::VecReal:
            return vec_real_node.erase(key);
        case NodeKind::VecString:
            return vec_string_node.erase(key);
    }

    return false;
}

bool Database::erase(const std::string& key)
{
    const auto iter = d_key_types.find(key);
    if (iter == d_key_types.end())
    {
        return false;
    }

    const auto erased = erase_from_node(key, iter->second);
    d_key_types.erase(iter);
    return erased;
}

} // namespace SimpleFluid
