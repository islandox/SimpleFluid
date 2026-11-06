#include "PitzDailyReplay.hh"
#include <iostream>
#include <functional>

int main()
{
    using namespace SimpleFluid::pitz_replay;
    const std::string header = "x,y,z,ux,uy,uz,pressure,k,epsilon,nu_t\n";
    const std::string valid = "0.05,0,0,1,2,3,4,0.1,0.2,0.01\n";
    auto require_failure = [&](const std::string& data)
    {
        Rows rows; std::istringstream input(data);
        try { read(input, rows, "test"); }
        catch (const std::exception&) { return; }
        throw std::runtime_error("Invalid replay input was accepted");
    };
    Rows rows; std::istringstream input(header + valid); read(input, rows, "test");
    if (rows.size() != 1 || rows.at(key(.05, 0, 0))[3] != 1) return 1;
    require_failure("");
    require_failure("x,y,z\n0,0,0\n");
    require_failure(header + valid + valid);
    require_failure(header + "0.05,0,0,1,2,3,4,nan,0.2,0.01\n");
    require_failure(header + "0.05,0,0,1,2,3,4,0.1foo,0.2,0.01\n");
    require_failure(header + "0.05,0,0,1,2,3,4,-0.1,0.2,0.01\n");
    require_failure(header + "0.05,0,0,1,2,3,4,0.1,0.2\n");
    require_failure(header + "0.05,0,0,1,2,3,4,0.1,0.2,0.01,0\n");
    // Duplicates across original partitions must also fail.
    std::istringstream duplicate(header + valid);
    try { read(duplicate, rows, "second rank"); return 2; } catch (const std::exception&) {}
    std::cout << "Replay parser checks passed\n";
}
