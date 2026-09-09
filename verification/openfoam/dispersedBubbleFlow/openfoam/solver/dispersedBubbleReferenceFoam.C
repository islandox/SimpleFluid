// Independent OpenFOAM FV reference for the constant-slip microbubble limit.
// Production is applied after implicit Euler/upwind transport, matching the
// documented operator splitting in the SimpleFluid model.
#include "fvCFD.H"
#include "StructuredCaseMesh.H"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    Foam::argList::addOption("steps","N","Run only N steps for a partial smoke check");
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    const StructuredCaseMesh grid(mesh);
    IOdictionary properties(IOobject("verificationProperties", runTime.constant(), mesh,
        IOobject::MUST_READ, IOobject::NO_WRITE));
    const auto parameter = [&](const word& name) { return readScalar(properties.lookup(name)); };
    const word mode(properties.lookup("mode"));
    const scalar height = parameter("height"), width = parameter("width");
    const scalar dt = parameter("dt"), speed = parameter("carrier_velocity") + parameter("slip_velocity");
    const scalar radius = parameter("nucleation_radius");
    // Isothermal IF97 reference liquid. Carrier flow and slip are prescribed,
    // so viscosity/conductivity are recorded but no momentum/energy is solved.
    const scalar temperature = parameter("temperature_K"), absolutePressure = parameter("absolute_pressure_Pa");
    const scalar rho = parameter("density_kg_m3"), cp = parameter("specific_heat_capacity_J_kg_K");
    const scalar mu = parameter("dynamic_viscosity_Pa_s"), conductivity = parameter("thermal_conductivity_W_m_K");
    const scalar sigma = parameter("surface_tension_N_m");
    const scalar bubbleVolume = 4.0 * constant::mathematical::pi / 3.0 * pow3(radius);
    const scalar molesPerBubble = bubbleVolume * (absolutePressure + 2 * sigma / radius)
        / (parameter("gas_constant") * temperature);
    const scalar source = mode == "steady" ? parameter("power_density") * parameter("yield_mol_per_j")
        * parameter("release_efficiency") : 0.0;
    const scalar initial = parameter(mode + "_initial_moles");
    const label cells = grid.cells();
    const scalar volume = height * sqr(width);
    const scalar volumeScale=volume;
    const scalar end = parameter(mode + "_end_time");
    const label nominalSteps = label(std::llround(end / dt));
    const label steps=args.getOrDefault<label>("steps",nominalSteps);
    if(steps<1||steps>nominalSteps)FatalErrorInFunction<<"Invalid smoke step limit"<<exit(FatalError);
    const label writeSteps = label(std::llround(parameter(mode + "_write_interval") / dt));
    if (returnReduce(mesh.nCells(), sumOp<label>()) != cells || writeSteps < 1 || mag(runTime.deltaTValue() - dt) > 1e-14)
        FatalErrorInFunction << "Unmatched mesh/time parameters" << exit(FatalError);
    // Programmatically created moments must retain processor coupling after
    // decomposePar; a uniform zeroGradient type would seal each partition.
    const wordList momentPatchTypes = StructuredCaseMesh::patch_types(mesh, "zeroGradient");
    volScalarField microMoles(IOobject("microMoles", runTime.timeName(), mesh,
        IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
        dimensionedScalar("initial", dimensionSet(0,-3,0,0,1,0,0), initial), momentPatchTypes);
    volScalarField microNumber(IOobject("microNumber", runTime.timeName(), mesh,
        IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
        dimensionedScalar("initial", dimless/dimVolume, initial/molesPerBubble), momentPatchTypes);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh.Sf() & dimensionedVector("bubbleVelocity", dimVelocity, vector(0,0,speed)));
    forAll(phi.boundaryField(), patch)
    {
        if (!mesh.boundary()[patch].coupled() && mesh.boundary()[patch].name() != "zmax")
            phi.boundaryFieldRef()[patch] = 0.0;
    }
    const label outlet = mesh.boundaryMesh().findPatchID("zmax");
    if (returnReduce(outlet < 0, orOp<bool>()))
        FatalErrorInFunction << "Missing outlet" << exit(FatalError);
    std::ofstream profiles((runTime.path()/"profiles.csv").c_str());
    std::ofstream history((runTime.path()/"history.csv").c_str());
    std::ofstream fields((runTime.path()/"fields.csv").c_str());
    if (returnReduce(!profiles.good() || !history.good() || !fields.good(), orOp<bool>()))
        FatalErrorInFunction << "Cannot create verification CSV files" << exit(FatalError);
    fields << std::setprecision(17)
        << "time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_s,uz_m_s\n";
    profiles << std::setprecision(17)
        << "time_s,sample,z_m,micro_moles_mol_m3,micro_number_m3,alpha_g,temperature_K,absolute_pressure_Pa,density_kg_m3,"
           "specific_heat_capacity_J_kg_K,dynamic_viscosity_Pa_s,thermal_conductivity_W_m_K,"
           "kinematic_viscosity_m2_s,thermal_diffusivity_m2_s,surface_tension_N_m,"
           "hydrogen_balance_mol,number_balance_relative\n";
    history << std::setprecision(17) << "time_s,inventory_mol,produced_mol,escaped_mol,outlet_mol_s,hydrogen_balance_mol,maximum_change_mol_m3\n";
    scalar escaped = 0, escapedNumber = 0, produced = 0, lastEscape = 0, maximumChange = 0;
    label steadyConsecutiveSteps = 0;
    auto writeCsv = [&](scalar time)
    {
        const scalar inventory = gSum(microMoles.primitiveField()*mesh.V());
        const scalar number = gSum(microNumber.primitiveField()*mesh.V());
        const scalar balance = inventory + escaped - initial*volume - produced;
        const scalar numberBalance = (number + escapedNumber - (initial*volume + produced)/molesPerBubble)
            / ((initial*volume + produced)/molesPerBubble);
        if (!std::isfinite(balance) || mag(balance) > 2e-13*volumeScale || mag(numberBalance) > 2e-8)
            FatalErrorInFunction << "Conservation gate failed" << exit(FatalError);
        forAll(microMoles, cell)
        {
            const auto c=mesh.C()[cell];
            const label ix=grid.interval(grid.x,c.x()),iy=grid.interval(grid.y,c.y());
            if(iy!=grid.ny()/2)continue;
            const scalar z = c.z();
            const label sample=grid.interval(grid.z,z);
            fields << time << ',' << sample*grid.nx()+ix << ',' << grid.x[ix] << ',' << grid.x[ix+1] << ',' << grid.z[sample] << ',' << grid.z[sample+1] << ','
                   << temperature << ',' << rho << ',' << microNumber[cell]*bubbleVolume
                   << ",0,0," << parameter("carrier_velocity") << '\n';
            if(ix!=grid.nx()/2)continue;
            profiles << time << ',' << sample << ',' << z << ',' << microMoles[cell] << ','
                     << microNumber[cell] << ',' << microNumber[cell]*bubbleVolume << ','
                     << temperature << ',' << absolutePressure << ',' << rho << ',' << cp << ','
                     << mu << ',' << conductivity << ',' << mu/rho << ',' << conductivity/(rho*cp) << ','
                     << sigma << ','
                     << balance << ',' << numberBalance << '\n';
        }
        history << time << ',' << inventory << ',' << produced << ',' << escaped << ','
                << lastEscape/dt << ',' << balance << ',' << maximumChange << '\n';
    };
    writeCsv(0);
    const auto loopWallStart = std::chrono::steady_clock::now();
    const auto loopCpuStart = std::clock();
    for (label step = 1; step <= steps; ++step)
    {
        ++runTime;
        if (mag(runTime.value()-step*dt) > 1e-11*max(scalar(1),end))
            FatalErrorInFunction << "Physical time does not match fixed step schedule" << exit(FatalError);
        const scalarField previous(microMoles.primitiveField());
        solve(fvm::ddt(microMoles) + fvm::div(phi, microMoles));
        solve(fvm::ddt(microNumber) + fvm::div(phi, microNumber));
        microMoles.correctBoundaryConditions();
        microNumber.correctBoundaryConditions();
        lastEscape = dt * gSum(phi.boundaryField()[outlet] * microMoles.boundaryField()[outlet]);
        escaped += lastEscape;
        escapedNumber += dt * gSum(phi.boundaryField()[outlet] * microNumber.boundaryField()[outlet]);
        microMoles.primitiveFieldRef() += source*dt;
        microNumber.primitiveFieldRef() += source*dt/molesPerBubble;
        produced += source*dt*volume;
        maximumChange = gMax(mag(microMoles.primitiveField() - previous));
        if (maximumChange < 1e-12 && mag(lastEscape/dt-source*volume) < 2e-11*volumeScale)
            ++steadyConsecutiveSteps;
        else steadyConsecutiveSteps = 0;
        const scalar minimumMoles = gMin(microMoles.primitiveField());
        const scalar minimumNumber = gMin(microNumber.primitiveField());
        if (minimumMoles < 0 || minimumNumber < 0)
            FatalErrorInFunction << "Negative microbubble moment" << exit(FatalError);
        if (step % writeSteps == 0 || step==steps) writeCsv(step*dt);
    }
    if (steps==nominalSteps && mode == "steady")
    {
        if (steadyConsecutiveSteps < 5)
            FatalErrorInFunction << "Steady convergence/outlet balance failed" << exit(FatalError);
        bool continuumPassed = true;
        forAll(microMoles, cell)
        {
            const scalar exact = source*mesh.C()[cell].z()/speed;
            const label iz=grid.interval(grid.z,mesh.C()[cell].z());
            const scalar dz=grid.z[iz+1]-grid.z[iz];
            const scalar truncation = source*(0.5*dz/speed+dt);
            continuumPassed = continuumPassed && mag(microMoles[cell]-exact) <= truncation*1.01+1e-12;
        }
        if (!returnReduce(continuumPassed, andOp<bool>()))
            FatalErrorInFunction << "Steady continuum profile failed" << exit(FatalError);
    }
    else if(steps==nominalSteps)
    {
        scalar l1 = 0;
        forAll(microMoles, cell)
        {
            const label sample=grid.interval(grid.z,mesh.C()[cell].z());
            const scalar dz=grid.z[sample+1]-grid.z[sample];
            const scalar upper=grid.z[sample+1];
            const scalar exact = initial*std::clamp((upper-speed*end)/dz, scalar(0), scalar(1));
            l1 += mag(microMoles[cell]-exact)*mesh.V()[cell]/(initial*volume);
        }
        reduce(l1, sumOp<scalar>());
        const scalar fraction=min(scalar(1),speed*end/height);
        if (l1 > 0.12/height || escaped < 0.8*fraction*initial*volume || escaped > min(scalar(1),1.2*fraction)*initial*volume)
            FatalErrorInFunction << "Transient translating-front/escape gate failed" << exit(FatalError);
    }
    profiles.flush(); history.flush(); fields.flush();
    const scalar loopWall = std::chrono::duration<scalar>(std::chrono::steady_clock::now()-loopWallStart).count();
    const scalar loopCpu = scalar(std::clock()-loopCpuStart)/CLOCKS_PER_SEC;
    std::ofstream performance((runTime.path()/"timing.json").c_str());
    performance << std::setprecision(17) << "{\"rank\":" << Pstream::myProcNo()
                << ",\"ranks\":" << Pstream::nProcs() << ",\"loop_wall_s\":" << loopWall
                << ",\"loop_cpu_s\":" << loopCpu << "}\n";
    profiles.flush(); history.flush(); fields.flush(); performance.flush();
    if (returnReduce(!profiles.good() || !history.good() || !fields.good() || !performance.good(), orOp<bool>()))
        FatalErrorInFunction << "CSV write failed" << exit(FatalError);
    Info << mode << " dispersed microbubble reference passed" << endl;
    if(steps<nominalSteps)Info<<"Partial smoke: "<<steps<<" of "<<nominalSteps<<" steps; not a complete comparison"<<nl;
    return 0;
}
