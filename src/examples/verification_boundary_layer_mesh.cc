/** Export native BoundaryLayerMeshFactory coordinates for both CFD solvers. */
#include "geometry/BoundaryLayerMeshFactory.hh"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        if (argc != 4)
            throw std::invalid_argument("Usage: verification_boundary_layer_mesh CASE uniform|medium|fine OUTPUT");
        const std::string name = argv[1], level = argv[2];
        if (level != "uniform" && level != "medium" && level != "fine")
            throw std::invalid_argument("Unknown mesh level");
        const bool fine = level == "fine", refine = level != "uniform";
        int nx = 1, nz = 8, count = 4;
        double width = 1, height = 1, depth = 1, thickness = .25, growth = 1.3;
        if (name == "bottomHeatedBubblyConvection")
        {
            nx = fine ? 36 : refine ? 24 : 12;
            nz = fine ? 48 : refine ? 32 : 16;
            width = height = .02;
            depth = .002;
            thickness = .0025;
            count = fine ? 8 : 6;
            growth = 1.25;
        }
        else if (name == "dispersedBubbleFlow")
        {
            nz = 40;
            thickness = .1;
            count = fine ? 8 : 6;
            growth = 1.3;
        }
        else if (name == "planarALE")
        {
            nz = fine ? 16 : 8;
            count = fine ? 6 : 4;
        }
        else
            throw std::invalid_argument("Unknown verification case");
        SimpleFluid::ArrReal x, z;
        for (int i = 0; i <= nx; ++i)
            x.push_back(width * i / nx);
        for (int i = 0; i <= nz; ++i)
            z.push_back(height * i / nz);
        SimpleFluid::Meshes::OrthogonalCartesian3D geometry({{x, {0, depth}, z}});
        const double first = thickness * (growth - 1) / (std::pow(growth, count) - 1);
        const double first_x = (name == "bottomHeatedBubblyConvection" ? width / 6 : thickness) * (growth - 1) /
                               (std::pow(growth, count) - 1);
        if (refine)
        {
            SimpleFluid::Arr<SimpleFluid::BoundaryLayerMeshFactory::BoundaryLayerSpec> specs;
            if (nx > 1)
                for (const char* face : {"xmin", "xmax"})
                    specs.push_back({face, static_cast<size_t>(count), first_x, growth});
            for (const char* face : {"zmin", "zmax"})
                specs.push_back({face, static_cast<size_t>(count), first, growth});
            SimpleFluid::BoundaryLayerMeshFactory(specs).build(geometry);
        }
        const std::filesystem::path output = argv[3];
        if (output.has_parent_path())
            std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        const auto& edges = geometry.cell_edges();
        stream << std::setprecision(17) << "# Native BoundaryLayerMeshFactory; " << name << " / " << level << "\n"
               << "# Layer count " << (refine ? count : 0) << ", first cell heights x=" << edges[0][1]-edges[0][0]
               << " z=" << edges[2][1]-edges[2][0]
               << ", growth " << growth << "; SI metres\n";
        size_t cells = 1;
        for (size_t axis = 0; axis < 3; ++axis)
        {
            stream << "xyz"[axis];
            for (double edge : edges[axis])
                stream << ' ' << edge;
            stream << '\n';
            cells *= edges[axis].size() - 1;
        }
        std::cout << name << ' ' << level << ": " << cells << " cells; wrote " << output << '\n';
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
