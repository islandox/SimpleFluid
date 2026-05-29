/**
 * @file DBNode.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Typed key-value node for the Database class.
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once


#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{
/**
 * @brief Key-value storage.
 *
 * A key stores T. Inserting a value with an existing
 * key replaces the previous value, including values of the other kind.
 *
 * @tparam T Type of values stored by the database.
 */
template <class T>
class DBNode
{
public:
    using key_type = std::string;
    using value_type = T;

    DBNode() = default;

    void set(const key_type& key, const value_type& value)
    {
        node.insert_or_assign(key, value);
    }

    void set(const key_type& key, value_type&& value)
    {
        node.insert_or_assign(key, std::move(value));
    }

    bool contains(const key_type& key) const
    {
        return node.contains(key);
    }

    value_type& get(const key_type& key)
    {
        auto iter = node.find(key);
        if (iter == node.end())
        {
            throw std::out_of_range("Database key not found: " + key);
        }

        return iter->second;
    }

    const value_type& get(const key_type& key) const
    {
        auto iter = node.find(key);
        if (iter == node.end())
        {
            throw std::out_of_range("Database key not found: " + key);
        }

        return iter->second;
    }

    bool erase(const key_type& key)
    {
        return node.erase(key) > 0;
    }

    void clear()
    {
        node.clear();
    }

    std::size_t size() const
    {
        return node.size();
    }

    bool empty() const
    {
        return node.empty();
    }
    
private:
    std::unordered_map<key_type, T> node;
};
} // namespace SimpleFluid
