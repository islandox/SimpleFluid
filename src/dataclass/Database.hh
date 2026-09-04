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
#include "DBNode.hh"

#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

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
        return d_values.contains(key);
    }

    bool erase(const std::string& key);

    void clear()
    {
        d_values.clear();
    }

    size_t size() const
    {
        return d_values.size();
    }

private:
    using DatabaseValue =
        std::variant<int, real_t, std::string, bool, ArrInt, ArrReal, ArrString>;

    template<class T>
    static constexpr bool is_supported_value_v =
        std::is_same_v<T, int> ||
        std::is_same_v<T, real_t> ||
        std::is_same_v<T, std::string> ||
        std::is_same_v<T, bool> ||
        std::is_same_v<T, ArrInt> ||
        std::is_same_v<T, ArrReal> ||
        std::is_same_v<T, ArrString>;

    std::unordered_map<std::string, DatabaseValue> d_values;
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
    static_assert(
        is_supported_value_v<Value>,
        "Unsupported Database value type");

    d_values.insert_or_assign(
        key,
        DatabaseValue{std::in_place_type<Value>, std::forward<T>(value)});
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
    static_assert(
        is_supported_value_v<T>,
        "Unsupported Database value type");

    auto iter = d_values.find(key);
    if (iter == d_values.end() || !std::holds_alternative<T>(iter->second))
    {
        throw std::out_of_range("Database key not found: " + key);
    }

    return std::get<T>(iter->second);
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
    static_assert(
        is_supported_value_v<T>,
        "Unsupported Database value type");

    auto iter = d_values.find(key);
    if (iter == d_values.end() || !std::holds_alternative<T>(iter->second))
    {
        throw std::out_of_range("Database key not found: " + key);
    }

    return std::get<T>(iter->second);
}

} // namespace SimpleFluid
