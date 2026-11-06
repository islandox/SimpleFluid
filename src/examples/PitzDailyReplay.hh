#pragma once

// Strict, partition-independent input for diagnostic replay of retained native cells.
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid::pitz_replay
{
using Row = std::array<double, 10>;
using Key = std::array<std::int64_t, 3>;
using Rows = std::map<Key, Row>;

inline std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> result;
    std::istringstream input(line);
    std::string item;
    while (std::getline(input, item, ','))
    {
        if (!item.empty() && item.back() == '\r') item.pop_back();
        result.push_back(item);
    }
    if (!line.empty() && line.back() == ',') result.emplace_back();
    return result;
}

inline Key key(double x, double y, double z)
{
    Key result{};
    const double xyz[] = {x, y, z};
    for (int i = 0; i < 3; ++i)
    {
        // Geometry matching tolerance, not a physical acceptance tolerance.
        const double scaled = xyz[i] * 1.0e12;
        if (!std::isfinite(scaled) || std::abs(scaled) >= 9.0e18)
            throw std::invalid_argument("Replay coordinate cannot be indexed.");
        result[i] = std::llround(scaled);
    }
    return result;
}

inline void read(std::istream& input, Rows& rows, const std::string& name)
{
    static const std::array<std::string, 10> required =
        {"x", "y", "z", "ux", "uy", "uz", "pressure", "k", "epsilon", "nu_t"};
    std::string line;
    if (!std::getline(input, line)) throw std::invalid_argument(name + ": missing header");
    const auto header = split(line);
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i)
        if (!columns.emplace(header[i], i).second)
            throw std::invalid_argument(name + ": duplicate header column");
    for (const auto& column : required)
        if (!columns.contains(column)) throw std::invalid_argument(name + ": missing " + column);
    size_t line_number = 1;
    while (std::getline(input, line))
    {
        ++line_number;
        const auto cells = split(line);
        if (cells.size() != header.size())
            throw std::invalid_argument(name + ": invalid row width at " + std::to_string(line_number));
        Row row{};
        for (size_t i = 0; i < row.size(); ++i)
        {
            const auto& value = cells.at(columns.at(required[i]));
            size_t consumed = 0;
            row[i] = std::stod(value, &consumed);
            if (consumed != value.size() || !std::isfinite(row[i]))
                throw std::invalid_argument(name + ": invalid numeric field at " + std::to_string(line_number));
        }
        if (row[7] <= 0 || row[8] <= 0 || row[9] < 0)
            throw std::invalid_argument(name + ": invalid turbulence state (no clipping is performed)");
        if (!rows.emplace(key(row[0], row[1], row[2]), row).second)
            throw std::invalid_argument(name + ": duplicate or geometrically ambiguous cell");
    }
    if (input.bad()) throw std::runtime_error(name + ": read failure");
}
} // namespace SimpleFluid::pitz_replay
