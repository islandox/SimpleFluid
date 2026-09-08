// Independent laminar momentum/pressure, energy, and dilute H2-moment reference.
#include "fvCFD.H"
#include "StructuredCaseMesh.H"
#include <cmath>
#include <fstream>
#include <iomanip>

int main(int argc, char* argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    const StructuredCaseMesh grid(mesh);
    if (Pstream::parRun()) FatalErrorInFunction << "Serial comparison fixture" << exit(FatalError);
    IOdictionary properties(IOobject("verificationProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    auto par=[&](const word& key){return readScalar(properties.lookup(key));};
    const label nx=grid.nx(),steps=label(std::llround(par("end_time")/par("dt")));
    const label stride=label(std::llround(par("write_interval")/par("dt")));
    const scalar width=par("width"),height=par("height"),dt=par("dt");
    const scalar rho0=par("density_kg_m3"),cp=par("specific_heat_capacity_J_kg_K"),mu=par("dynamic_viscosity_Pa_s");
    const scalar k=par("thermal_conductivity_W_m_K"),beta=par("thermal_expansion_1_K"),T0=par("temperature_K");
    const scalar sigma=par("surface_tension_N_m"),pressure=par("absolute_pressure_Pa"),R=par("gas_constant");
    const scalar rhoGas=pressure*par("hydrogen_molar_mass")/(R*T0),fourPi=4*constant::mathematical::pi/3;
    const dimensionedScalar nu("nu",dimViscosity,mu/rho0);
    const dimensionedScalar conductivity("k",dimEnergy/dimTime/dimLength/dimTemperature,k);
    volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
    surfaceScalarField phi(IOobject("phi",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),fvc::flux(U));
    volScalarField moles(IOobject("moles",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,
        dimensionedScalar(dimensionSet(0,-3,0,0,1,0,0),Zero),"zeroGradient");
    volScalarField number(IOobject("number",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,
        dimensionedScalar(dimless/dimVolume,Zero),"zeroGradient");
    volScalarField alpha(IOobject("alpha",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,
        dimensionedScalar(dimless,Zero),"zeroGradient");
    volScalarField rhoCp(IOobject("rhoCp",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,
        dimensionedScalar(dimEnergy/dimVolume/dimTemperature,rho0*cp),"zeroGradient");
    volScalarField q(IOobject("q",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,
        dimensionedScalar(dimEnergy/dimVolume/dimTime,Zero),"zeroGradient");
    volVectorField buoyancy(IOobject("buoyancy",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,
        dimensionedVector(dimAcceleration,Zero),"zeroGradient");
    forAll(q,cell)
    {
        const vector c=mesh.C()[cell];
        if(c.x()>width/3 && c.x()<2*width/3 && c.z()<height/8) q[cell]=par("power_density");
    }
    const scalar power=sum(q.primitiveField()*mesh.V());
    if(mag(power-par("power_density")*(width/3)*(height/8)*par("depth"))>1e-12)
        FatalErrorInFunction<<"Shared mesh must preserve source extent and power"<<exit(FatalError);
    const scalar volume=sum(mesh.V().field());
    const label outlet=mesh.boundaryMesh().findPatchID("zmax");
    scalar produced=0,escaped=0,thermalResidual=0;
    std::ofstream fields((runTime.path()/"fields.csv").c_str()),history((runTime.path()/"history.csv").c_str());
    fields.exceptions(std::ios::badbit|std::ios::failbit);history.exceptions(std::ios::badbit|std::ios::failbit);
    fields<<std::setprecision(17)<<"time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_s,uz_m_s\n";
    history<<std::setprecision(17)<<"time_s,sample,temperature_max_K,temperature_mean_K,alpha_max,speed_max_m_s,uz_min_m_s,uz_max_m_s,hydrogen_mol,produced_mol,escaped_mol,hydrogen_balance_mol,heat_power_W,continuity_per_s,thermal_step_residual_J\n";
    auto check=[&](bool valid,const char* message){if(!valid)FatalErrorInFunction<<message<<exit(FatalError);};
    auto write=[&](label step)
    {
        scalar tmax=T0,tmean=0,amax=0,umax=0,uzmin=0,uzmax=0;
        forAll(T,cell)
        {
            const vector c=mesh.C()[cell];const label ix=grid.interval(grid.x,c.x()),iz=grid.interval(grid.z,c.z());
            tmax=max(tmax,T[cell]);tmean+=T[cell]*mesh.V()[cell]/volume;amax=max(amax,alpha[cell]);
            umax=max(umax,mag(U[cell]));uzmin=min(uzmin,U[cell].z());uzmax=max(uzmax,U[cell].z());
            check(std::isfinite(T[cell])&&T[cell]>290&&T[cell]<320&&std::isfinite(mag(U[cell]))&&alpha[cell]>=0&&alpha[cell]<0.02,
                "Outside dilute reference-water envelope");
            fields<<runTime.value()<<','<<iz*nx+ix<<','<<grid.x[ix]<<','<<grid.x[ix+1]<<','<<grid.z[iz]<<','<<grid.z[iz+1]<<','
                  <<T[cell]<<','<<rho0*(1-beta*(T[cell]-T0))<<','<<alpha[cell]<<','<<U[cell].x()<<','<<U[cell].y()<<','<<U[cell].z()<<'\n';
        }
        const scalar inventory=sum(moles.primitiveField()*mesh.V()),balance=inventory+escaped-produced;
        const scalar continuity=gMax(mag(fvc::div(phi)().primitiveField()));
        check(mag(balance)<1e-13&&continuity<1e-6,"Hydrogen/continuity gate failed");
        check(mag(thermalResidual)<1e-6,"Discrete thermal step budget failed");
        history<<runTime.value()<<",global,"<<tmax<<','<<tmean<<','<<amax<<','<<umax<<','<<uzmin<<','<<uzmax<<','
               <<inventory<<','<<produced<<','<<escaped<<','<<balance<<','<<power<<','<<continuity<<','<<thermalResidual<<'\n';
        Info<<"step="<<step<<" t="<<runTime.value()<<" Tmax="<<tmax<<" alpha="<<amax<<" Umax="<<umax<<" uz=["<<uzmin<<','<<uzmax<<']'<<nl;
        if(step==steps)check(tmax-T0>0.05&&amax>1e-6&&uzmin< -1e-5&&uzmax>1e-5,"No heated bubbly plume and return flow");
    };
    write(0);
    for(label step=1;step<=steps;++step)
    {
        ++runTime;
        check(mag(runTime.value()-step*dt)<1e-10,"Time schedule mismatch");
        forAll(T,cell)
        {
            const scalar rho=rho0*(1-beta*(T[cell]-T0))*(1-alpha[cell])+rhoGas*alpha[cell];
            rhoCp[cell]=rho*cp;
            buoyancy[cell]=vector(0,0,-par("gravity")*(rho-rho0)/rho0);
        }
        rhoCp.correctBoundaryConditions();buoyancy.correctBoundaryConditions();
        fvVectorMatrix UEqn(fvm::ddt(U)+fvm::div(phi,U)-fvm::laplacian(nu,U)==buoyancy);
        solve(UEqn == -fvc::grad(p));
        for(label corrector=0;corrector<3;++corrector)
        {
            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(),U,p));
            surfaceScalarField phiHbyA(fvc::flux(HbyA)+fvc::interpolate(rAU)*fvc::ddtCorr(U,phi));
            adjustPhi(phiHbyA,U,p);constrainPressure(p,U,phiHbyA,rAU);
            fvScalarMatrix pEqn(fvm::laplacian(rAU,p)==fvc::div(phiHbyA));
            pEqn.setReference(0,0);pEqn.solve();
            phi=phiHbyA-pEqn.flux();
            U=HbyA-rAU*fvc::grad(p);U.correctBoundaryConditions();
        }
        const scalarField oldT(T.primitiveField());
        const surfaceScalarField capacityFlux(fvc::flux(phi,rhoCp));
        solve(rhoCp*fvm::ddt(T)+fvm::div(capacityFlux,T)-fvm::laplacian(conductivity,T)==q);
        T.correctBoundaryConditions();
        thermalResidual=sum(rhoCp.primitiveField()*(T.primitiveField()-oldT)*mesh.V())-power*dt;
        forAll(T.boundaryField(),patch)
            thermalResidual-=dt*k*sum(T.boundaryField()[patch].snGrad()*mesh.magSf().boundaryField()[patch]);
        surfaceScalarField bubblePhi(phi+(mesh.Sf()&dimensionedVector("slip",dimVelocity,vector(0,0,par("slip_velocity")))));
        forAll(bubblePhi.boundaryField(),patch)
            if(patch!=outlet)bubblePhi.boundaryFieldRef()[patch]=0;
            else bubblePhi.boundaryFieldRef()[patch]=max(bubblePhi.boundaryField()[patch],scalar(0));
        solve(fvm::ddt(moles)+fvm::div(bubblePhi,moles));
        solve(fvm::ddt(number)+fvm::div(bubblePhi,number));
        moles.correctBoundaryConditions();number.correctBoundaryConditions();
        escaped+=dt*sum(bubblePhi.boundaryField()[outlet]*moles.boundaryField()[outlet]);
        forAll(T,cell)
        {
            const scalar t=T[cell],G=par("yield_molecules_per_100_ev");
            const scalar LET=(-1.3387e-6*t-3.4319e-5)*par("uranium_concentration")-6.6431e-3*t+8.8142;
            const scalar waterRadius=(-2.862e-15*t*t+7.3996e-13*t-9.9925e-11)*LET*LET
                +(8.7909e-14*t*t-9.7928e-13*t+3.4558e-9)*LET+9.7683e-14*t*t-4.0125e-11*t+4.9092e-9;
            const scalar rn=(5.165e-5-1.732e-3+0.02245-0.1554+1.134)*(0.3554+0.4264*G-0.0400*G*G)*waterRadius;
            const scalar nb=fourPi*(pressure*pow3(rn)+2*sigma*sqr(rn))/(R*t);
            check(rn>1e-12&&rn<1e-3,"Nucleation radius outside supported range");
            const scalar source=q[cell]*par("yield_mol_per_j")*dt;
            moles[cell]+=source;number[cell]+=source/nb;
            produced+=source*mesh.V()[cell];
            if(moles[cell]>0&&number[cell]>0)
            {
                const scalar target=moles[cell]*R*t/(number[cell]*fourPi);
                scalar lo=0,hi=std::cbrt(target/pressure);
                for(label iter=0;iter<64;++iter)
                {
                    const scalar r=0.5*(lo+hi);
                    if(pressure*pow3(r)+2*sigma*sqr(r)>target)hi=r;else lo=r;
                }
                alpha[cell]=fourPi*number[cell]*pow3(0.5*(lo+hi));
            }
            else alpha[cell]=0;
        }
        moles.correctBoundaryConditions();number.correctBoundaryConditions();alpha.correctBoundaryConditions();
        if(step%stride==0||step==steps)write(step);
    }
    U.write();T.write();alpha.write();p.write();
    return 0;
}
