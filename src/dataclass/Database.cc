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

/**
 * @brief Erase a key-value pair from the database regardless of type.
 *
 * @param key Key to erase from the database.
 * @return true if the key was found and erased, false otherwise.
 */
bool Database::erase(const std::string& key)
{
    return d_values.erase(key) > 0;
}

} // namespace SimpleFluid
