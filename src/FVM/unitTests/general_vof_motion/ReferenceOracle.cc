#include "ReferenceOracle.hh"
#include <algorithm>
#include <cmath>
#include <utility>

namespace SimpleFluid::VoF::test::motion
{
// Test-local monotone hull, half-plane clipping and shoelace integration. Native
// points supply the cell shape only; no production PLIC cuts/measures form the oracle.
long double cross(XY a, XY b, XY c)
{
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
Polygon hull(Polygon p)
{
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    Polygon q;
    for (const auto& a : p)
    {
        while (q.size() > 1 && cross(q[q.size() - 2], q.back(), a) <= 0)
            q.pop_back();
        q.push_back(a);
    }
    const auto lower = q.size();
    for (auto i = p.rbegin() + 1; i != p.rend(); ++i)
    {
        while (q.size() > lower && cross(q[q.size() - 2], q.back(), *i) <= 0)
            q.pop_back();
        q.push_back(*i);
    }
    q.pop_back();
    return q;
}
long double area(const Polygon& p)
{
    long double twice = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        twice += p[i][0] * p[(i + 1) % p.size()][1] - p[i][1] * p[(i + 1) % p.size()][0];
    return std::abs(twice) / 2;
}
long double clipped_area(const Polygon& p, long double time, bool gas)
{
    Polygon cut;
    const auto distance = [&](XY a)
    {
        const auto d = a[0] + 2 * speed * time * a[1] - (.5L + speed * time);
        return gas ? d : -d;
    };
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const auto a = p[i], b = p[(i + 1) % p.size()];
        const auto da = distance(a), db = distance(b);
        if (da <= 0)
            cut.push_back(a);
        if ((da < 0 && db > 0) || (da > 0 && db < 0))
        {
            const auto t = da / (da - db);
            cut.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])});
        }
    }
    return area(cut);
}
Polygon intersection(Polygon p, const Polygon& clip)
{
    for (std::size_t j = 0; j < clip.size() && !p.empty(); ++j)
    {
        Polygon next;
        const auto a = clip[j], b = clip[(j + 1) % clip.size()];
        for (std::size_t i = 0; i < p.size(); ++i)
        {
            const auto u = p[i], v = p[(i + 1) % p.size()];
            const auto du = cross(a, b, u), dv = cross(a, b, v);
            if (du >= 0)
                next.push_back(u);
            if ((du < 0 && dv > 0) || (du > 0 && dv < 0))
            {
                const auto f = du / (du - dv);
                next.push_back({u[0] + f * (v[0] - u[0]), u[1] + f * (v[1] - u[1])});
            }
        }
        p = std::move(next);
    }
    return p;
}
} // namespace SimpleFluid::VoF::test::motion
