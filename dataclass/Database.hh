/**
 * @file Database.hh
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "typedefs.hh"

#include "DBNode.hh"

namespace SimpleFluid
{

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
        return int_node.contains(key) || real_node.contains(key) ||
               string_node.contains(key) || bool_node.contains(key) ||
               vec_int_node.contains(key) || vec_real_node.contains(key);
    }

    bool erase(const std::string& key)
    {
        return int_node.erase(key) || real_node.erase(key) ||
               string_node.erase(key) || bool_node.erase(key) ||
               vec_int_node.erase(key) || vec_real_node.erase(key);
    }

    void clear()
    {
        int_node.clear();
        real_node.clear();
        string_node.clear();
        bool_node.clear();
        vec_int_node.clear();
        vec_real_node.clear();
    }

    size_t size() const
    {
        return int_node.size() + real_node.size() + string_node.size() +
               bool_node.size() + vec_int_node.size() + vec_real_node.size();
    }

private:
    DBNode<int> int_node;
    DBNode<real_t> real_node;
    DBNode<std::string> string_node;
    DBNode<bool> bool_node;

    DBNode<std::vector<int>> vec_int_node;
    DBNode<std::vector<real_t>> vec_real_node;
};

template <class T>
void Database::set(const std::string& key, T&& value)
{
    if constexpr (std::same_as<T, int>)
    {
        int_node.set(key, value);
    }
    else if constexpr (std::same_as<T, real_t>)
    {
        real_node.set(key, value);
    }
    else if constexpr (std::same_as<T, std::string>)
    {
        string_node.set(key, value);
    }
    else if constexpr (std::same_as<T, bool>)
    {
        bool_node.set(key, value);
    }
    else if constexpr (std::same_as<T, std::vector<int>>)
    {
        vec_int_node.set(key, value);
    }
    else if constexpr (std::same_as<T, std::vector<real_t>>)
    {
        vec_real_node.set(key, value);
    }
    else
    {
        static_assert(always_false<T>, "Unsupported type for Database::set");
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
    else
    {
        static_assert(always_false<T>, "Unsupported type for Database::get");
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
    else
    {
        static_assert(always_false<T>, "Unsupported type for Database::get");
    }
}

} // namespace SimpleFluid
