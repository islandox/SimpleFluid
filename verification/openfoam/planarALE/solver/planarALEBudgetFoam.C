// Independent OpenFOAM finite-volume reference for uniform thermal expansion.
// The fluid and mesh share the affine normal velocity: relative flux is zero.
#include "fvCFD.H"
#include "StructuredCaseMesh.H"

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>

int main(int argc, char* argv[])
{
    Foam::argList::addOption("mode", "steady|transient", "Heating history or source-off equilibrium");
    Foam::argList::addOption("steps","N","Run only N steps for a partial smoke check");
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    const StructuredCaseMesh grid(mesh);
    const word mode(args.getOrDefault<word>("mode", "transient"));
    if (mode != "steady" && mode != "transient")
    {
        FatalErrorInFunction << "Use -mode steady|transient" << exit(FatalError);
    }
    const IOdictionary water
    (
        IOobject("referenceWater", runTime.constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE)
    );
    const scalar rho0 = readScalar(water.lookup("density_kg_m3"));
    const scalar cp = readScalar(water.lookup("specific_heat_capacity_J_kg_K"));
    const scalar beta = readScalar(water.lookup("thermal_expansion_1_K"));
    const scalar T0 = readScalar(water.lookup("temperature_K"));
    const scalar conductivity = readScalar(water.lookup("thermal_conductivity_W_m_K"));
    const scalar viscosity = readScalar(water.lookup("dynamic_viscosity_Pa_s"));
    const scalar absolutePressure = readScalar(water.lookup("absolute_pressure_Pa"));
    const scalar initialHeight=grid.z.last()-grid.z.first();
    const scalar area=(grid.x.last()-grid.x.first())*(grid.y.last()-grid.y.first());
    const scalar initialVolume=area*initialHeight,initialMass=rho0*initialVolume;
    const scalar volumeScale=initialVolume,lengthScale=initialHeight;
    const scalar dt = 1.0, heating = 4.0e5;
    const label heatedSteps = 20, quietSteps = 5;
    const label nominalSteps = heatedSteps + (mode == "steady" ? quietSteps : 0);
    const label steps=args.getOrDefault<label>("steps",nominalSteps);
    if(steps<1||steps>nominalSteps||(args.found("steps")&&mode=="steady"))
        FatalErrorInFunction<<"Invalid transient smoke step limit"<<exit(FatalError);
    runTime.setDeltaT(dt);
    const pointField referencePoints(mesh.points());
    const scalarField cellMass(rho0 * mesh.V().field());
    volScalarField T(IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh);
    T = dimensionedScalar("T0", dimTemperature, T0);
    T.correctBoundaryConditions();
    const dimensionedScalar thermalConductivity
    (
        "thermalConductivity", dimEnergy/dimTime/dimLength/dimTemperature, conductivity
    );
    volScalarField rhoCp
    (
        IOobject("rhoCp", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
        dimensionedScalar("rhoCp", dimEnergy/dimVolume/dimTemperature, rho0*cp),
        StructuredCaseMesh::patch_types(mesh, "zeroGradient")
    );
    surfaceScalarField relativeHeatCapacityFlux
    (
        IOobject("relativeHeatCapacityFlux", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh, dimensionedScalar(dimEnergy/dimTemperature/dimTime, Zero)
    );
    T.oldTime();
    rhoCp.oldTime();
    std::ofstream csv((runTime.path()/"history.csv").c_str());
    std::ofstream spatial((runTime.path()/"fields.csv").c_str());
    spatial.exceptions(std::ios::badbit | std::ios::failbit);
    spatial << std::setprecision(17)
        << "time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_s,uz_m_s\n";
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(17)
        << "time_s,sample,temperature_K,level_m,volume_m3,liquid_mass_kg,energy_J,cumulative_heat_J,"
           "mass_residual_kg,energy_balance_residual_J,gcl_residual_m3_per_s,"
           "analytic_temperature_error_K,analytic_level_error_m,density_kg_m3,cp_J_kg_K,mu_Pa_s,k_W_m_K,"
           "nu_m2_s,thermal_diffusivity_m2_s,thermal_expansion_1_K,absolute_pressure_Pa,relative_flux_max_m3_s,relative_flux_per_volume_s\n";
    scalar level = initialHeight, exactT = T0, cumulativeHeat = 0, previousT = T0, previousLevel = initialHeight;
    label quietCount = 0;
    auto check = [](scalar residual, scalar tolerance, const char* name)
    {
        if (!std::isfinite(residual) || mag(residual) > tolerance)
        {
            FatalErrorInFunction << name << " residual=" << residual << ", tolerance=" << tolerance
                << exit(FatalError);
        }
    };
    const auto loopStart = std::chrono::steady_clock::now();
    const auto cpuStart = std::clock();
    for (label step = 0; step <= steps; ++step)
    {
        const scalar q = step <= heatedSteps ? heating : 0.0;
        if (step)
        {
            ++runTime;
            T.oldTime();
            rhoCp.oldTime();
            bool converged = false;
            // Picard coupling of actual moved geometry, conservative FV energy,
            // thermal liquid EOS, and sum(Mcell/rhoLiquid). No analytic solution
            // or SimpleFluid-produced state enters this iteration.
            for (label corrector = 0; corrector < 30; ++corrector)
            {
                pointField trialPoints(referencePoints);
                forAll(trialPoints, pointi)
                    trialPoints[pointi].z() *= level/initialHeight;
                mesh.movePoints(trialPoints);
                forAll(cellMass, celli)
                    rhoCp[celli] = cellMass[celli]*cp/mesh.V()[celli];
                rhoCp.correctBoundaryConditions();
                fvScalarMatrix energyEquation
                (
                    fvm::ddt(rhoCp, T) + fvm::div(relativeHeatCapacityFlux, T)
                  - fvm::laplacian(thermalConductivity, T)
                 == dimensionedScalar("q", dimEnergy/dimVolume/dimTime, q)
                );
                energyEquation.solve();
                T.correctBoundaryConditions();
                scalar targetLevel = 0;
                bool validDensity = true;
                forAll(cellMass, celli)
                {
                    const scalar liquidDensity = rho0*(1-beta*(T[celli]-T0));
                    if (liquidDensity <= 1 || !std::isfinite(liquidDensity))
                        validDensity = false;
                    else
                        targetLevel += cellMass[celli]/liquidDensity/area;
                }
                reduce(validDensity, andOp<bool>());
                if (!validDensity)
                    FatalErrorInFunction << "Invalid thermal liquid density" << exit(FatalError);
                // Every rank uses the same liquid-volume target. The affine
                // motion then agrees at shared points and at convergence.
                reduce(targetLevel, sumOp<scalar>());
                if (mag(targetLevel-level) <= 1e-13*lengthScale)
                {
                    converged = true;
                    break;
                }
                level = targetLevel;
            }
            if (!converged)
                FatalErrorInFunction << "ALE thermal/geometry coupling did not converge" << exit(FatalError);
            // Independent analytic BE oracle, used exclusively for acceptance.
            const scalar a = 1-beta*(exactT-T0), b = q*dt/(rho0*cp);
            exactT += 2*b/(a+std::sqrt(a*a-4*beta*b));
        }
        scalar volume = 0, mass = 0, energy = 0, gcl = 0, temperatureError = 0;
        long double energyAccumulator=0;
        forAll(cellMass, celli)
        {
            volume += mesh.V()[celli];
            mass += rhoCp[celli]/cp*mesh.V()[celli];
            energyAccumulator+=static_cast<long double>(rhoCp[celli])*mesh.V()[celli]*T[celli];
            const scalar z = mesh.C()[celli].z();
            const auto c=mesh.C()[celli];
            const label ix=grid.interval(grid.x,c.x()),iy=grid.interval(grid.y,c.y());
            const scalar cellArea=(grid.x[ix+1]-grid.x[ix])*(grid.y[iy+1]-grid.y[iy]);
            const scalar dz = mesh.V()[celli]/cellArea;
            const label sample=grid.interval(grid.z,z*initialHeight/level)*grid.nx()+ix;
            // Kinematic affine reference only: this application solves no momentum.
            const scalar uz = step ? z/level*(level-previousLevel)/dt : 0.0;
            if(iy==grid.ny()/2) spatial << runTime.value() << ',' << sample << ',' << grid.x[ix] << ',' << grid.x[ix+1] << ',' << z-0.5*dz << ',' << z+0.5*dz << ','
                    << T[celli] << ',' << rho0*(1-beta*(T[celli]-T0)) << ",0,0,0," << uz << '\n';
            const scalar error = T[celli]-exactT;
            temperatureError = max(temperatureError,
                std::isfinite(error) ? mag(error) : std::numeric_limits<scalar>::infinity());
        }
        energy=scalar(energyAccumulator);
        reduce(volume, sumOp<scalar>());
        reduce(mass, sumOp<scalar>());
        reduce(energy, sumOp<scalar>());
        reduce(temperatureError, maxOp<scalar>());
        check(temperatureError, 2e-7, "Cell temperature analytic error");
        if (step)
        {
            cumulativeHeat += q*volume*dt;
            const volScalarField meshDivergence(fvc::div(mesh.phi()));
            forAll(cellMass, celli)
                gcl = max(gcl, mag((mesh.V()[celli]-mesh.V0()[celli])/dt
                    - meshDivergence[celli]*mesh.V()[celli]));
        }
        reduce(gcl, maxOp<scalar>());
        const scalar temperature = energy/(mass*cp);
        const scalar exactLevel = initialHeight/(1-beta*(exactT-T0));
        const scalar density = rho0*(1-beta*(temperature-T0));
        const scalar massResidual = mass-initialMass;
        const scalar energyResidual = energy-initialMass*cp*T0-cumulativeHeat;
        check(massResidual, 2e-10*volumeScale, "Liquid mass conservation");
        check(energyResidual, 5e-5*volumeScale, "Liquid energy conservation");
        check(gcl, 2e-11, "Mesh GCL");
        check(level-exactLevel, 5e-10*lengthScale, "Level analytic error");
        check(volume-area*level, 2e-11*volumeScale, "Mesh volume and level closure");
        if (step > heatedSteps)
        {
            check(temperature-previousT, 2e-8, "Steady temperature change");
            check(level-previousLevel, 2e-11*lengthScale, "Steady level change");
            ++quietCount;
        }
        check(runTime.value()-step*dt, 1e-13, "Accepted physical time");
        csv << runTime.value() << ",global," << temperature << ',' << level << ',' << volume << ',' << mass << ','
            << energy << ',' << cumulativeHeat << ',' << massResidual << ',' << energyResidual << ',' << gcl
            << ',' << temperature-exactT << ',' << level-exactLevel << ',' << density << ',' << cp << ','
            << viscosity << ',' << conductivity << ',' << viscosity/density << ',' << conductivity/(density*cp)
            << ',' << beta << ',' << absolutePressure << ",0,0\n";
        previousT = temperature;
        previousLevel = level;
    }
    csv.flush();
    spatial.flush();
    const scalar loopWall = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-loopStart).count();
    const scalar loopCpu = double(std::clock()-cpuStart)/CLOCKS_PER_SEC;
    std::ofstream timing((runTime.path()/"timing.json").c_str());
    timing.exceptions(std::ios::badbit | std::ios::failbit);
    timing << std::setprecision(17) << "{\"rank\":" << Pstream::myProcNo()
        << ",\"ranks\":" << Pstream::nProcs() << ",\"local_cells\":" << mesh.nCells()
        << ",\"loop_wall_s\":" << loopWall << ",\"loop_cpu_s\":" << loopCpu << "}\n";
    if (mode == "steady" && quietCount != quietSteps)
        FatalErrorInFunction << "Steady state did not pass five consecutive source-off steps" << exit(FatalError);
    T.write();
    rhoCp.write();
    mesh.write();
    Info<< "planarALE " << mode << ": " << steps << " accepted steps, " << quietCount
        << " source-off convergence checks; wrote history.csv" << nl;
    if(steps<nominalSteps)Info<<"Partial smoke: "<<steps<<" of "<<nominalSteps<<" steps; not a complete comparison"<<nl;
    return 0;
}
