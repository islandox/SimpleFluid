/**
 * @file Database.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Simple typed key-value database for mesh and solver configuration.
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

    template<class T>
    DBNode<T>& node();

    template<class T>
    const DBNode<T>& node() const;

    template<class T>
    static consteval NodeKind node_kind();

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

/**
 * @brief Store a value under a given key, replacing any previous value.
 *
 * @tparam T Type of the value to store.
 * @param key Key to associate with the value.
 * @param value Value to store.
 */
template <class T>
void Database::set(const std::string& key, T&& value)
{
    using Value = std::remove_cvref_t<T>;

    erase(key);
    node<Value>().set(key, std::forward<T>(value));
    d_key_types[key] = node_kind<Value>();
}

/**
 * @brief Retrieve a mutable reference to a value by key.
 *
 * @tparam T Expected type of the stored value.
 * @param key Key to look up.
 * @return Mutable reference to the stored value.
 * @throws std::out_of_range if the key is missing or the type does not match.
 */
template <class T>
T& Database::get(const std::string& key)
{
    return node<T>().get(key);
}

/**
 * @brief Retrieve a const reference to a value by key.
 *
 * @tparam T Expected type of the stored value.
 * @param key Key to look up.
 * @return Const reference to the stored value.
 * @throws std::out_of_range if the key is missing or the type does not match.
 */
template <class T>
const T& Database::get(const std::string& key) const
{
    return node<T>().get(key);
}

template<class T>
DBNode<T>& Database::node()
{
    if constexpr (std::same_as<T, int>)
        return int_node;
    else if constexpr (std::same_as<T, real_t>)
        return real_node;
    else if constexpr (std::same_as<T, std::string>)
        return string_node;
    else if constexpr (std::same_as<T, bool>)
        return bool_node;
    else if constexpr (std::same_as<T, std::vector<int>>)
        return vec_int_node;
    else if constexpr (std::same_as<T, std::vector<real_t>>)
        return vec_real_node;
    else if constexpr (std::same_as<T, std::vector<std::string>>)
        return vec_string_node;
    else
        static_assert(
            utils::TMP::always_false_v<T>,
            "Unsupported Database value type");
}

template<class T>
const DBNode<T>& Database::node() const
{
    return const_cast<Database*>(this)->node<T>();
}

template<class T>
consteval auto Database::node_kind() -> NodeKind
{
    if constexpr (std::same_as<T, int>)
        return NodeKind::Int;
    else if constexpr (std::same_as<T, real_t>)
        return NodeKind::Real;
    else if constexpr (std::same_as<T, std::string>)
        return NodeKind::String;
    else if constexpr (std::same_as<T, bool>)
        return NodeKind::Bool;
    else if constexpr (std::same_as<T, std::vector<int>>)
        return NodeKind::VecInt;
    else if constexpr (std::same_as<T, std::vector<real_t>>)
        return NodeKind::VecReal;
    else if constexpr (std::same_as<T, std::vector<std::string>>)
        return NodeKind::VecString;
    else
        static_assert(
            utils::TMP::always_false_v<T>,
            "Unsupported Database value type");
}

} // namespace SimpleFluid
