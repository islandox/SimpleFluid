/** @file FaceTransfer.cc
 * @brief Shared fixed-width face transfer framing.
 */
#include "solvers/details/GeneralVoF/FaceTransfer.hh"
namespace SimpleFluid::detail::general_vof
{
void append_face_word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned a = 0; a < 8; ++a)
        bytes.push_back(char(value >> (8 * a)));
}
std::uint64_t read_face_word(const std::vector<char>& bytes, std::size_t offset)
{
    std::uint64_t value = 0;
    for (unsigned a = 0; a < 8; ++a)
        value |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + a))) << (8 * a);
    return value;
}
} // namespace SimpleFluid::detail::general_vof
