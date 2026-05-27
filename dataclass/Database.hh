/**
 * @file Database.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "typedefs.hh"
#include "utils/TMP_helpers.hh"

#include "DBNode.hh"

#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Simple typed key-value database.
 *
 * Stores values of several predefined types and provides type-safe accessors.
 */
class Database
{
public:
    Database() = default;

    template <class T>
    void set(const std::string& key, T&& value);

    template <class T>
    T& get(const std::string& key);

    template <class T>
    const T& get(const std::string& key) const;

    bool contains(const std::string& key) const
    {
        return d_key_types.contains(key);
    }

    bool erase(const std::string& key);

    void clear()
    {
        int_node.clear();
        real_node.clear();
        string_node.clear();
        bool_node.clear();
        vec_int_node.clear();
        vec_real_node.clear();
        vec_string_node.clear();
        d_key_types.clear();
    }

    size_t size() const
    {
        return d_key_types.size();
    }

private:
    enum class NodeKind : uint8_t
    {
        Int,
        Real,
        String,
        Bool,
        VecInt,
        VecReal,
        VecString
    };

    bool erase_from_node(const std::string& key, NodeKind kind);

    DBNode<int> int_node;
    DBNode<real_t> real_node;
    DBNode<std::string> string_node;
    DBNode<bool> bool_node;

    DBNode<std::vector<int>> vec_int_node;
    DBNode<std::vector<real_t>> vec_real_node;
    DBNode<std::vector<std::string>> vec_string_node;

    std::unordered_map<std::string, NodeKind> d_key_types;
};

inline bool Database::erase_from_node(const std::string& key, NodeKind kind)
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

inline bool Database::erase(const std::string& key)
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

template <class T>
void Database::set(const std::string& key, T&& value)
{
    using Value = std::remove_cvref_t<T>;

    erase(key);

    if constexpr (std::same_as<Value, int>)
    {
        int_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::Int;
    }
    else if constexpr (std::same_as<Value, real_t>)
    {
        real_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::Real;
    }
    else if constexpr (std::same_as<Value, std::string>)
    {
        string_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::String;
    }
    else if constexpr (std::same_as<Value, bool>)
    {
        bool_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::Bool;
    }
    else if constexpr (std::same_as<Value, std::vector<int>>)
    {
        vec_int_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::VecInt;
    }
    else if constexpr (std::same_as<Value, std::vector<real_t>>)
    {
        vec_real_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::VecReal;
    }
    else if constexpr (std::same_as<Value, std::vector<std::string>>)
    {
        vec_string_node.set(key, std::forward<T>(value));
        d_key_types[key] = NodeKind::VecString;
    }
    else
    {
        static_assert(utils::always_false_v<Value>, "Unsupported type for Database::set");
    }
}

template <class T>
T& Database::get(const std::string& key)
{
    if constexpr (std::same_as<T, int>)
    {
        return int_node.get(key);
    }   
    else if constexpr (std::same_as<T, real_t>)
    {
        return real_node.get(key);
    }
    else if constexpr (std::same_as<T, std::string>)
    {
        return string_node.get(key);
    }
    else if constexpr (std::same_as<T, bool>)
    {
        return bool_node.get(key);
    }
    else if constexpr (std::same_as<T, std::vector<int>>)
    {
        return vec_int_node.get(key);
    }
    else if constexpr (std::same_as<T, std::vector<real_t>>)
    {
        return vec_real_node.get(key);
    }
    else if constexpr (std::same_as<T, std::vector<std::string>>)
    {
        return vec_string_node.get(key);
    }
    else
    {
        static_assert(utils::always_false_v<T>, "Unsupported type for Database::get");
    }
}

template <class T>
const T& Database::get(const std::string& key) const
{
    if constexpr (std::same_as<T, int>)
    {
        return int_node.get(key);
    }
    else if constexpr (std::same_as<T, real_t>)
    {
        return real_node.get(key);
    }
    else if constexpr (std::same_as<T, std::string>)
    {
        return string_node.get(key);
    }
    else if constexpr (std::same_as<T, bool>)
    {
        return bool_node.get(key);
    }
    else if constexpr (std::same_as<T, std::vector<int>>)
    {
        return vec_int_node.get(key);
    }
    else if constexpr (std::same_as<T, std::vector<real_t>>)
    {
        return vec_real_node.get(key);
    }
    else if constexpr (std::same_as<T, std::vector<std::string>>)
    {
        return vec_string_node.get(key);
    }
    else
    {
        static_assert(utils::always_false_v<T>, "Unsupported type for Database::get");
    }
}

} // namespace SimpleFluid
