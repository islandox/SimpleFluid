/**
 * @file ConservativeInterface.cc
 * @author islandox
 * @brief Scale-aware convex polygon coverage validation for conservative seams.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#include "geometry/mesh/ConservativeInterface.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace SimpleFluid::Meshes
{
namespace
{
struct Point { real_t x, y; };
real_t cross(Point a, Point b, Point p) { return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x); }
real_t signed_area(const std::vector<Point>& p)
{
    real_t area=0;
    for(size_t i=0;i<p.size();++i) area+=p[i].x*p[(i+1)%p.size()].y-p[i].y*p[(i+1)%p.size()].x;
    return area/2;
}
std::vector<Point> intersect(std::vector<Point> polygon, const std::vector<Point>& clip)
{
    for(size_t edge=0;edge<clip.size() && !polygon.empty();++edge)
    {
        const auto a=clip[edge],b=clip[(edge+1)%clip.size()];
        std::vector<Point> result;
        for(size_t i=0;i<polygon.size();++i)
        {
            const auto p=polygon[i],q=polygon[(i+1)%polygon.size()];
            const auto dp=cross(a,b,p),dq=cross(a,b,q);
            if(dp>=0) result.push_back(p);
            if((dp<0)!=(dq<0))
            {
                const auto t=dp/(dp-dq);
                result.push_back({p.x+t*(q.x-p.x),p.y+t*(q.y-p.y)});
            }
        }
        polygon=std::move(result);
    }
    return polygon;
}
}
void validate_planar_subdivision(const InterfaceFacePolygon& coarse,
    std::span<const InterfaceFacePolygon> fine, real_t absolute_length, real_t relative)
{
    if(!std::isfinite(absolute_length) || absolute_length<0 || !std::isfinite(relative) || relative<=0)
        throw std::invalid_argument("Invalid planar subdivision tolerance.");
    if(coarse.vertices.size()<3 || fine.empty() || !(coarse.area>0) || !std::isfinite(coarse.area))
        throw std::invalid_argument("A nonconforming interface requires a positive planar coarse face and fine tiles.");
    const auto normal=coarse.area_vector/coarse.area;
    if(!std::isfinite(normal.norm()) || std::abs(normal.norm()-1)>relative)
        throw std::invalid_argument("Coarse face area vector disagrees with its area.");
    const auto origin=coarse.vertices[0];
    const auto direction=coarse.vertices[1]-origin;
    if(!(direction.norm()>0) || !std::isfinite(direction.norm()))
        throw std::invalid_argument("Degenerate coarse face edge.");
    const auto u=direction/direction.norm(), v=normal.cross(u);
    real_t scale=std::sqrt(coarse.area);
    for(const auto p:coarse.vertices) scale=std::max(scale,(p-origin).norm());
    const auto length_tol=absolute_length+relative*scale;
    const auto area_tol=absolute_length*scale+relative*coarse.area;
    auto project=[&](const InterfaceFacePolygon& face, bool is_coarse)
    {
        if(face.vertices.size()<3 || !std::isfinite(face.area) || !(face.area>0)
            || (face.area_vector/face.area-normal*(is_coarse?1.0:-1.0)).norm()>relative)
            throw std::invalid_argument("Coarse/fine faces require opposite outward normals and positive areas.");
        std::vector<Point> points;
        for(const auto p:face.vertices)
        {
            const auto d=p-origin;
            if(!std::isfinite(d.norm()) || std::abs(d.dot(normal))>length_tol)
                throw std::invalid_argument("Nonconforming interface faces must lie on one plane.");
            points.push_back({d.dot(u),d.dot(v)});
        }
        auto area=signed_area(points);
        if(area<0) { std::reverse(points.begin(),points.end()); area=-area; }
        if(!(area>0) || std::abs(area-face.area)>area_tol)
            throw std::invalid_argument("Planar face polygon area disagrees with native face geometry.");
        for(size_t i=0;i<points.size();++i)
            for(const auto point:points)
                if(cross(points[i],points[(i+1)%points.size()],point)<-area_tol)
                    throw std::invalid_argument("Nonconforming interface polygons must be convex.");
        real_t cx=0,cy=0;
        for(size_t i=0;i<points.size();++i)
        {
            const auto a=points[i],b=points[(i+1)%points.size()];
            const auto c=a.x*b.y-a.y*b.x;
            cx+=(a.x+b.x)*c; cy+=(a.y+b.y)*c;
        }
        const auto centroid=origin+u*(cx/(6*area))+v*(cy/(6*area));
        if((centroid-face.centroid).norm()>length_tol)
            throw std::invalid_argument("Planar face centroid disagrees with its vertex loop.");
        return points;
    };
    const auto coarse_polygon=project(coarse,true);
    std::vector<std::vector<Point>> tiles;
    real_t total=0;
    for(const auto& face:fine)
    {
        auto polygon=project(face,false);
        const auto inside=std::abs(signed_area(intersect(polygon,coarse_polygon)));
        if(std::abs(inside-face.area)>area_tol)
            throw std::invalid_argument("Fine face extends outside its coarse face.");
        for(const auto& previous:tiles)
            if(std::abs(signed_area(intersect(polygon,previous)))>area_tol)
                throw std::invalid_argument("Fine interface tiles overlap.");
        total+=face.area;
        tiles.push_back(std::move(polygon));
    }
    if(std::abs(total-coarse.area)>area_tol)
        throw std::invalid_argument("Fine interface tiles leave a gap or cover the coarse face more than once.");
}
} // namespace SimpleFluid::Meshes
