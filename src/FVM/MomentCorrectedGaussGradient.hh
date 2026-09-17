/** @file MomentCorrectedGaussGradient.hh @brief Linearity-preserving Gauss-linear reconstruction. */
#pragma once
#include "FVM/details/VectorLaplacian.hh"

namespace SimpleFluid::FVM
{
/**
 * Gauss-linear face interpolation with an inverse coordinate-moment correction.
 * Inputs must be ghost-synchronized; this function performs only local work.
 * M=sum(S_f d_f^T)/V and b=sum(S_f (phi_f-phi_P))/V, then M grad(phi)=b.
 * This preserves affine fields on skewed/curved cells and constant offsets.
 * Components is 1 for a scalar and 3 for a Cartesian vector; output is row-major.
 */
template<size_t Components, class Field, class Gradient, class MeshType,
         class BoundaryConditionProvider, class BoundaryValueProvider>
void moment_corrected_gauss_cell_gradient(const Field& field,
    BoundaryConditionProvider boundary_condition, BoundaryValueProvider boundary_value,
    Gradient& gradient, const TransportGeometryCache<MeshType>& cache)
{
    const auto& mesh=field.mesh(); cache.require_mesh(mesh);
    if (&gradient.mesh()!=&mesh) throw std::invalid_argument("Gauss gradient fields must use the cached mesh.");
    const auto& geometry=cache.assembly_geometry();
    const auto& locations=cache.boundary_locations();
    const auto values=field.local_read_view(); auto result=gradient.owned_write_view();
    using lo=typename MeshType::local_ordinal_type;
    using scalar=typename MeshType::scalar_type;
    for (size_t owned=0; owned<mesh.num_owned_cells(); ++owned)
    {
        const auto cell=static_cast<lo>(owned);
        const long double volume=geometry.volumes[owned];
        if (!std::isfinite(volume) || volume<=0) throw std::invalid_argument("Gauss gradient requires positive finite volume.");
        std::array<std::array<long double,3>,3> moment{};
        std::array<std::array<long double,Components>,3> rhs{};
        for (auto slot=geometry.face_offsets[owned]; slot<geometry.face_offsets[owned+1]; ++slot)
        {
            const auto& face=geometry.faces[slot]; const auto f=face.face_lid;
            typename MeshType::Vec3 direction{};
            std::array<long double,Components> increment{};
            if (face.interior)
            {
                const auto dp=mesh.cell_to_face_distance(f,cell), dn=mesh.cell_to_face_distance(f,face.other);
                if (!std::isfinite(dp+dn) || dp+dn<=0) throw std::invalid_argument("Gauss face distances are invalid.");
                const auto weight=dp/(dp+dn);
                direction=mesh.cell_center_vector(f,cell)*weight;
                for (size_t c=0;c<Components;++c) increment[c]=weight*(values(face.other,c)-values(cell,c));
            }
            else
            {
                const auto normal=mesh.face_normal_outward(f,cell);
                const auto distance=detail::boundary_normal_distance(mesh,f,cell);
                if (!std::isfinite(distance) || distance<=0) throw std::invalid_argument("Gauss boundary distance is invalid.");
                direction=normal*distance; // Untagged exterior defaults to homogeneous Neumann.
                if (locations[f].active)
                {
                    const auto& loc=locations[f];
                    const auto condition=boundary_condition(loc.batch_id,loc.in_batch_id);
                    const auto type=[&]
                    { if constexpr (requires { condition.type; }) return condition.type; else return condition; }();
                    if (type==BoundaryConditionType::Slip)
                    {
                        static_assert(Components==1 || Components==3);
                        if constexpr (Components==3)
                        {
                            long double un=0;
                            for(size_t c=0;c<3;++c) un+=normal.component(c)*values(cell,c);
                            for(size_t c=0;c<3;++c) increment[c]=-normal.component(c)*un;
                        }
                        else throw std::invalid_argument("Scalar Gauss gradients do not have a slip condition.");
                    }
                    else if (type==BoundaryConditionType::Neumann)
                    {
                        if constexpr (Components==1) increment[0]=condition.value*distance;
                        else
                        {
                            const auto value=boundary_value(loc.batch_id,loc.in_batch_id);
                            for(size_t c=0;c<3;++c) increment[c]=value.component(c)-values(cell,c);
                        }
                    }
                    else if (type==BoundaryConditionType::Dirichlet || type==BoundaryConditionType::NoSlip)
                    {
                        direction=mesh.face_centroid(f)-mesh.cell_centroid(cell);
                        const auto value=boundary_value(loc.batch_id,loc.in_batch_id);
                        if constexpr (Components==1) increment[0]=value-values(cell,0);
                        else for(size_t c=0;c<3;++c) increment[c]=value.component(c)-values(cell,c);
                    }
                    else throw std::invalid_argument("Gauss periodic conditions require connected topology.");
                }
            }
            const auto area=detail::laplacian_face_area(mesh,f,cell);
            for(size_t i=0;i<3;++i)
            {
                for(size_t j=0;j<3;++j) moment[i][j]+=static_cast<long double>(area.component(i))*direction.component(j)/volume;
                for(size_t c=0;c<Components;++c) rhs[i][c]+=area.component(i)*increment[c]/volume;
            }
        }
        long double scale=0;
        for(const auto& row:moment) for(auto value:row) scale=std::max(scale,std::abs(value));
        // Nonsymmetric moments require a pivoted solve, not a symmetric pseudoinverse.
        for(size_t pivot=0;pivot<3;++pivot)
        {
            size_t best=pivot;
            for(size_t row=pivot+1;row<3;++row)
                if(std::abs(moment[row][pivot])>std::abs(moment[best][pivot])) best=row;
            if(!std::isfinite(scale) || scale==0 || std::abs(moment[best][pivot])<=64*std::numeric_limits<scalar>::epsilon()*scale)
                throw std::invalid_argument("Gauss coordinate moments are singular or ill-conditioned.");
            std::swap(moment[pivot],moment[best]); std::swap(rhs[pivot],rhs[best]);
            const auto diagonal=moment[pivot][pivot];
            for(auto& value:moment[pivot]) value/=diagonal;
            for(auto& value:rhs[pivot]) value/=diagonal;
            for(size_t row=0;row<3;++row) if(row!=pivot)
            {
                const auto factor=moment[row][pivot];
                for(size_t j=0;j<3;++j) moment[row][j]-=factor*moment[pivot][j];
                for(size_t c=0;c<Components;++c) rhs[row][c]-=factor*rhs[pivot][c];
            }
        }
        for(size_t c=0;c<Components;++c) for(size_t i=0;i<3;++i)
        {
            const auto value=static_cast<scalar>(rhs[i][c]);
            if(!std::isfinite(value)) throw std::invalid_argument("Gauss gradient is nonfinite.");
            result(cell,c*3+i)=value;
        }
    }
}
} // namespace SimpleFluid::FVM
