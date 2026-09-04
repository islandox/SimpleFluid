/**
 * @file DatabaseOptionReader.hh
 * @brief Context-aware typed access to model configuration databases.
 */
#pragma once

#include "dataclass/Database.hh"

#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::detail
{

/**
 * @brief Read model options while presenting one consistent error contract.
 *
 * Successful reads retain the Database exact-type semantics. Missing optional
 * values use their caller-provided defaults, while missing required values and
 * type mismatches are reported as context-rich `std::invalid_argument` errors.
 */
class DatabaseOptionReader
{
public:
    /** @brief Bind an option reader to a database and diagnostic context. */
    DatabaseOptionReader(const Database& database, std::string context)
        : d_database(database), d_context(std::move(context))
    {
    }

    /** @brief Return whether an option is present. */
    [[nodiscard]] bool contains(const std::string& key) const { return d_database.contains(key); }

    /** @brief Read an optional value or return @p fallback when absent. */
    template<class T> [[nodiscard]] T value_or(const std::string& key, T fallback) const
    {
        if (!contains(key))
        {
            return fallback;
        }
        return read<T>(key);
    }

    /** @brief Read a required value. */
    template<class T> [[nodiscard]] T required(const std::string& key) const
    {
        if (!contains(key))
        {
            throw std::invalid_argument(d_context + " option '" + key + "' is required.");
        }
        return read<T>(key);
    }

private:
    template<class T> [[nodiscard]] T read(const std::string& key) const
    {
        try
        {
            return d_database.get<T>(key);
        }
        catch (const std::out_of_range&)
        {
            throw std::invalid_argument(d_context + " option '" + key + "' has the wrong type.");
        }
    }

    const Database& d_database;
    std::string d_context;
};

} // namespace SimpleFluid::detail
