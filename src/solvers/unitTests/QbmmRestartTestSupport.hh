#pragma once

#include <Teuchos_CommHelpers.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid::Testing::QbmmRestart
{
inline void append_restart_word(std::vector<std::byte>& body, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
        body.push_back(static_cast<std::byte>((value >> shift) & 0xff));
}

inline void append_restart_checksums(const Teuchos::Comm<int>& comm, std::vector<std::byte>& body)
{
    std::uint64_t local = 14695981039346656037ULL, image = 0;
    for (const auto byte : body)
        local = (local ^ std::to_integer<std::uint64_t>(byte)) * 1099511628211ULL;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &image);
    append_restart_word(body, local);
    append_restart_word(body, image);
}

inline std::string restart_path(const Teuchos::Comm<int>& comm, const char* prefix, const char* stage)
{
    return std::string(prefix) + ".rank" + std::to_string(comm.getRank()) + stage;
}

inline void require_io(const Teuchos::Comm<int>& comm, bool valid)
{
    int local = valid, global = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("Could not read or write a rank-local QBMM restart fixture.");
}

inline void write_restart(
    const Teuchos::Comm<int>& comm, const char* prefix, const char* stage, std::span<const std::byte> bytes)
{
    std::ofstream file(restart_path(comm, prefix, stage), std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    require_io(comm, bool(file));
}

inline std::vector<std::byte> read_restart(const Teuchos::Comm<int>& comm, const char* prefix, const char* stage)
{
    std::ifstream file(restart_path(comm, prefix, stage), std::ios::binary | std::ios::ate);
    const std::streamoff size = file.tellg();
    require_io(comm, size >= 0);
    std::vector<std::byte> bytes(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require_io(comm, bool(file));
    return bytes;
}
} // namespace SimpleFluid::Testing::QbmmRestart
