#pragma once
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace SimpleFluid::detail::general_vof
{
/** Opt-in, rank-local replay evidence. No collectives or acceptance policy. */
template<class Write> void flow_trace(int rank, const char* event, Write&& write) noexcept
{
    try
    {
        const auto* path = std::getenv("SIMPLEFLUID_VOF_FLOW_TRACE");
        if (!path || !*path)
            return;
        std::ostringstream line;
        line.imbue(std::locale::classic());
        line << std::setprecision(21) << "{\"event\":\"" << event << "\",\"rank\":" << rank;
        write(line);
        line << "}\n";
        std::ofstream output(std::string(path) + ".rank-" + std::to_string(rank) + ".jsonl", std::ios::app);
        output << line.str();
    }
    catch (...)
    {
        // Optional diagnostic I/O cannot introduce a rank-local exception
        // before the solver's next collective or change numerical admission.
    }
}
} // namespace SimpleFluid::detail::general_vof
