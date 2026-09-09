/** Shared explicit Cartesian cell edges for paired verification cases. */
#pragma once
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace SimpleFluid::Verification
{
struct VerificationMesh
{
    ArrReal x, y, z;
    size_t nx() const { return x.size() - 1; }
    size_t ny() const { return y.size() - 1; }
    size_t nz() const { return z.size() - 1; }
    size_t cells() const { return nx() * ny() * nz(); }
    auto coordinates() const -> Vec3D<ArrReal> { return {{x, y, z}}; }
    static size_t interval(const ArrReal& edges, double center)
    {
        const auto it = std::upper_bound(edges.begin(), edges.end(), center);
        if (it == edges.begin() || it == edges.end())
            throw std::runtime_error("Cell centre lies outside reference mesh");
        return static_cast<size_t>(it - edges.begin() - 1);
    }
};

inline VerificationMesh read_verification_mesh(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot read verification mesh: " + path.string());
    std::map<std::string, ArrReal> axes;
    for (std::string line; std::getline(input, line);)
    {
        line = line.substr(0, line.find('#'));
        std::istringstream stream(line);
        std::string axis;
        if (!(stream >> axis))
            continue;
        if ((axis != "x" && axis != "y" && axis != "z") || axes.contains(axis))
            throw std::runtime_error("Invalid or duplicate mesh axis");
        ArrReal edges;
        double value;
        while (stream >> value)
        {
            if (!std::isfinite(value) || (!edges.empty() && value <= edges.back()))
                throw std::runtime_error("Invalid mesh edge ordering");
            edges.push_back(value);
        }
        if (!stream.eof() || edges.size() < 2)
            throw std::runtime_error("Malformed verification mesh edges");
        axes.emplace(axis, std::move(edges));
    }
    if (axes.size() != 3)
        throw std::runtime_error("Verification mesh requires x/y/z cell edges");
    return {axes.at("x"), axes.at("y"), axes.at("z")};
}
} // namespace SimpleFluid::Verification
