/** @file algebraic_motion_qualification.hh
 * @brief Independent manufactured-motion oracle and opt-in algebraic VoF qualification campaign.
 */
// Included inside testGeneralVoFMotionAccuracy.cc's anonymous namespace, after
// its independent hull/area oracle. This is test-only manufactured transport.
namespace algebraic_motion
{
using XYZ = std::array<long double, 3>;
constexpr long double pi = 3.141592653589793238462643383279502884L;
enum class Motion
{
    PlanarPatch,
    AxisDrop,
    DiagonalDrop,
    FullRotation,
    ShearReturn,
    ThreeDimensional
};
enum class Mesh
{
    Uniform,
    Skew,
    CoarseFine
};
struct Case
{
    Motion motion;
    Mesh mesh;
    GeneralVoFTransportBackend backend;
    double courant;
    bool temporal = false;
};
const char* name(Motion motion)
{
    switch (motion)
    {
        case Motion::PlanarPatch:
            return "PlanarPatch";
        case Motion::AxisDrop:
            return "AxisDrop";
        case Motion::DiagonalDrop:
            return "DiagonalDrop";
        case Motion::FullRotation:
            return "FullRotation";
        case Motion::ShearReturn:
            return "ShearReturn";
        case Motion::ThreeDimensional:
            return "ThreeDimensional";
    }
    throw std::logic_error("Unknown manufactured motion");
}
long double mesh_shear(const Case& c)
{
    return c.mesh == Mesh::Uniform ? 0 : .25L;
}
XYZ translation(Motion m)
{
    if (m == Motion::ThreeDimensional)
        return {.12L, .08L, .1L};
    return {.2L, m == Motion::AxisDrop ? 0.L : .1L, 0.L};
}
Polygon reference_polygon(Motion m, long double t)
{
    Polygon p;
    const bool planar = m == Motion::PlanarPatch, rotation = m == Motion::FullRotation;
    const XY center = rotation ? XY{.65L, .5L} : m == Motion::ShearReturn ? XY{.5L, .5L} : XY{.35L, .4L};
    const unsigned vertices = planar ? 4 : 128;
    for (unsigned i = 0; i < vertices; ++i)
    {
        long double x, y;
        if (planar)
        {
            const XY q[4]{{-.14L, -.055L}, {.14L, -.055L}, {.14L, .055L}, {-.14L, .055L}};
            x = q[i][0] * std::cos(.37L) - q[i][1] * std::sin(.37L);
            y = q[i][0] * std::sin(.37L) + q[i][1] * std::cos(.37L);
        }
        else
        {
            x = (rotation ? .105L : .12L) * std::cos(2 * pi * i / vertices);
            y = (rotation ? .07L : .12L) * std::sin(2 * pi * i / vertices);
        }
        x += center[0];
        y += center[1];
        if (rotation)
        {
            const auto a = x - .5L, b = y - .5L;
            x = .5L + a * std::cos(2 * pi * t) - b * std::sin(2 * pi * t);
            y = .5L + a * std::sin(2 * pi * t) + b * std::cos(2 * pi * t);
        }
        else if (m == Motion::ShearReturn)
            x += .8L * std::min(t, 1 - t) * (y - .5L);
        else
        {
            const auto v = translation(m);
            x += t * v[0];
            y += t * v[1];
        }
        p.push_back({x, y});
    }
    return p;
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
struct Cell
{
    Polygon xy;
    long double bottom, top;
    XYZ center;
    long double width;
    long double volume() const { return area(xy) * (top - bottom); }
};
Cell cell(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& geometry, long double s)
{
    Polygon p;
    long double bottom = 2, top = -1;
    long double xmin = 2, xmax = -1, ymin = 2, ymax = -1;
    for (const auto& q : geometry->points())
    {
        const long double x = q.value[0] - s * q.value[1], y = q.value[1];
        p.push_back({x, y});
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
        ymin = std::min(ymin, y);
        ymax = std::max(ymax, y);
        bottom = std::min(bottom, static_cast<long double>(q.value[2]));
        top = std::max(top, static_cast<long double>(q.value[2]));
    }
    const auto center = geometry->centroid();
    // Perpendicular width of a sheared X-slab, rather than its diagonal extent.
    return {hull(std::move(p)), bottom, top, {center[0], center[1], center[2]},
        std::min({(xmax - xmin) / std::sqrt(1 + s * s), ymax - ymin, top - bottom})};
}
long double phase(const Cell& c, Motion m, long double t, const Polygon& polygon)
{
    long double bottom = 0, top = 1;
    if (m == Motion::ThreeDimensional)
    {
        bottom = .3L + .1L * t;
        top = .55L + .1L * t;
    }
    const auto height = std::max(0.L, std::min(c.top, top) - std::max(c.bottom, bottom));
    return area(intersection(c.xy, polygon)) * height;
}
long double exact_volume(Motion m)
{
    return area(reference_polygon(m, 0)) * (m == Motion::ThreeDimensional ? .25L : 1.L);
}
long double extension_f(long double coordinate)
{
    if (coordinate < .125L)
        return 8 * coordinate;
    if (coordinate > .875L)
        return 8 * (1 - coordinate);
    return 1;
}
long double extension_g(long double coordinate)
{
    if (coordinate < .125L)
        return -3 * coordinate;
    if (coordinate > .875L)
        return 3 * (1 - coordinate);
    return coordinate - .5L;
}
// The transition has F'=-pi*(1-3t^2+2t^3)<=0. Its curl is
// tangential rotation with a sign fixed by the logical coordinate midplanes.
long double rotation_potential(long double radius2)
{
    constexpr long double inner2 = .3L * .3L, outer2 = .45L * .45L;
    if (radius2 <= inner2)
        return -pi * radius2;
    if (radius2 >= outer2)
        return -pi * (inner2 + (outer2 - inner2) / 2);
    const auto t = (radius2 - inner2) / (outer2 - inner2);
    return -pi * (inner2 + (outer2 - inner2) * (t - t * t * t + .5L * t * t * t * t));
}
// Vector-potential one-form in logical coordinates. A_X=0 also makes this
// the physical one-form under x=X+sY. Piece boundaries coincide with every
// campaign grid. Face-normal signs cannot reverse inside a logical grid face.
XYZ potential(XYZ p, Motion m, long double t)
{
    const auto x = p[0] - .5L, y = p[1] - .5L;
    if (m == Motion::FullRotation)
        return {0, 0, rotation_potential(x * x + y * y)};
    if (m == Motion::ShearReturn)
        return {0, 0, (t < .5L ? .4L : -.4L) * extension_g(p[1]) * extension_g(p[1]) * extension_f(p[0])};
    const auto v = translation(m);
    return {0, m == Motion::ThreeDimensional ? .1L * extension_g(p[0]) * extension_f(p[2]) : 0,
        v[0] * extension_f(p[0]) * extension_g(p[1]) - v[1] * extension_g(p[0]) * extension_f(p[1])};
}
long double edge_integral(XYZ a, XYZ b, Motion m, long double t)
{
    // Six-point Gauss integrates each polynomial piece (degree <= 8) exactly
    // in real arithmetic. Split at every cutoff breakpoint before integrating.
    constexpr long double nodes[6]{-.9324695142031520278123L, -.6612093864662645136614L, -.2386191860831969086305L,
        .2386191860831969086305L, .6612093864662645136614L, .9324695142031520278123L};
    constexpr long double weights[6]{.1713244923791703450403L, .3607615730481386075698L, .4679139345726910473899L,
        .4679139345726910473899L, .3607615730481386075698L, .1713244923791703450403L};
    std::vector<long double> breaks{0, 1};
    for (unsigned d = 0; d < 3; ++d)
        if (a[d] != b[d])
            for (const auto boundary : {.125L, .875L})
            {
                const auto f = (boundary - a[d]) / (b[d] - a[d]);
                if (f > 0 && f < 1)
                    breaks.push_back(f);
            }
    if (m == Motion::FullRotation)
    {
        const auto dx = b[0] - a[0], dy = b[1] - a[1];
        const auto x = a[0] - .5L, y = a[1] - .5L;
        const auto quadratic = dx * dx + dy * dy;
        const auto linear = 2 * (x * dx + y * dy);
        if (quadratic > 0)
            for (const auto radius2 : {.3L * .3L, .45L * .45L})
            {
                const auto constant = x * x + y * y - radius2;
                const auto discriminant = linear * linear - 4 * quadratic * constant;
                if (discriminant >= 0)
                    for (const auto sign : {-1.L, 1.L})
                    {
                        const auto root = (-linear + sign * std::sqrt(discriminant)) / (2 * quadratic);
                        if (root > 0 && root < 1)
                            breaks.push_back(root);
                    }
            }
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    long double result = 0;
    for (std::size_t i = 1; i < breaks.size(); ++i)
        for (unsigned g = 0; g < 6; ++g)
        {
            const auto half = (breaks[i] - breaks[i - 1]) / 2;
            const auto f = (breaks[i] + breaks[i - 1]) / 2 + half * nodes[g];
            XYZ p;
            for (unsigned d = 0; d < 3; ++d)
                p[d] = a[d] + f * (b[d] - a[d]);
            const auto v = potential(p, m, t);
            for (unsigned d = 0; d < 3; ++d)
                result += half * weights[g] * v[d] * (b[d] - a[d]);
        }
    return result;
}
void validate_single_stream_face(
    std::vector<XYZ> points, Motion motion, long double shear_value, long double time, std::uint64_t face_gid)
{
    const auto reject = [&](const char* reason)
    { throw std::invalid_argument(std::string(reason) + " face=" + std::to_string(face_gid)); };
    if (points.size() < 3)
        reject("Manufactured fixture has insufficient face geometry");
    for (auto& p : points)
        p[0] -= shear_value * p[1];
    XYZ normal{};
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto a = points[i], b = points[(i + 1) % points.size()];
        for (unsigned d = 0; d < 3; ++d)
            normal[d] += a[(d + 1) % 3] * b[(d + 2) % 3] - a[(d + 2) % 3] * b[(d + 1) % 3];
    }
    if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
        reject("Manufactured fixture face normal is unresolved");
    const bool horizontal =
        std::all_of(points.begin(), points.end(), [&](const auto& p) { return p[2] == points.front()[2]; });
    if (horizontal && motion != Motion::ThreeDimensional)
        return; // Identically zero normal velocity.
    XYZ center{};
    for (const auto& p : points)
        for (unsigned d = 0; d < 3; ++d)
            center[d] += p[d] / points.size();
    if (motion != Motion::FullRotation)
        for (unsigned d = 0; d < (motion == Motion::ThreeDimensional ? 3u : 2u); ++d)
        {
            long double lo = points.front()[d], hi = lo;
            for (const auto& p : points)
            {
                lo = std::min(lo, p[d]);
                hi = std::max(hi, p[d]);
            }
            for (const auto boundary : {.125L, .875L})
                if (lo < boundary && boundary < hi)
                    reject("Manufactured fixture face crosses an extension-box interior");
        }
    // Derivatives are selected from this face's interior box, including their
    // one-sided limits at box-boundary vertices. No derivative is sampled at a
    // discontinuity and no tolerance converts an uncertain sign to acceptance.
    const auto fp = [](long double coordinate) { return coordinate < .125L ? 8.L : coordinate > .875L ? -8.L : 0.L; };
    const auto gp = [](long double coordinate) { return coordinate < .125L || coordinate > .875L ? -3.L : 1.L; };
    const XYZ fprime{fp(center[0]), fp(center[1]), fp(center[2])};
    const XYZ gprime{gp(center[0]), gp(center[1]), gp(center[2])};
    const auto normal_velocity = [&](XYZ p)
    {
        XYZ velocity{};
        if (motion == Motion::ShearReturn)
        {
            const auto sign = time < .5L ? 1 : -1;
            velocity[0] = sign * .8L * extension_g(p[1]) * gprime[1] * extension_f(p[0]);
            velocity[1] = -sign * .4L * extension_g(p[1]) * extension_g(p[1]) * fprime[0];
        }
        else
        {
            const auto v = translation(motion);
            velocity[0] = v[0] * extension_f(p[0]) * gprime[1] - v[1] * extension_g(p[0]) * fprime[1];
            velocity[1] = -v[0] * fprime[0] * extension_g(p[1]) + v[1] * gprime[0] * extension_f(p[1]);
            if (motion == Motion::ThreeDimensional)
            {
                velocity[0] -= .1L * extension_g(p[0]) * fprime[2];
                velocity[2] = .1L * gprime[0] * extension_f(p[2]);
            }
        }
        long double result = 0;
        for (unsigned d = 0; d < 3; ++d)
            result += normal[d] * velocity[d];
        return result;
    };
    bool positive = false, negative = false;
    const auto examine = [&](long double value)
    {
        if (!std::isfinite(value))
            reject("Manufactured fixture normal sign is unresolved");
        positive = positive || value > 0;
        negative = negative || value < 0;
        if (positive && negative)
            reject("manufactured fixture requires spatial multistream");
    };
    if (horizontal)
    {
        // Genuine 3D w is affine (constant on an aligned box); extrema occur
        // at vertices of the convex horizontal native polygon.
        for (const auto& p : points)
            examine(normal_velocity(p));
        return;
    }
    if (normal[2] != 0)
        reject("Manufactured fixture requires a horizontal or vertical native face");
    XYZ a = points.front(), b = a;
    long double maximum_distance = 0;
    for (const auto& p : points)
        for (const auto& q : points)
        {
            const auto distance = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
            if (distance > maximum_distance)
            {
                a = p;
                b = q;
                maximum_distance = distance;
            }
        }
    if (!(maximum_distance > 0))
        reject("Manufactured fixture vertical face chord is unresolved");
    for (const auto& p : points)
        if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) != 0)
            reject("Manufactured fixture vertical face chord is unresolved");
    const auto at = [&](long double t) { return XYZ{a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), center[2]}; };
    if (motion == Motion::FullRotation)
    {
        // F' is negative only inside r<.45. Restrict the chord first; outside
        // it the normal velocity is exactly zero. Its remaining sign factor
        // nX*(Y-.5)-nY*(X-.5) is linear, so endpoint limits are authoritative.
        const auto dx = b[0] - a[0], dy = b[1] - a[1];
        const auto x = a[0] - .5L, y = a[1] - .5L;
        const auto linear = 2 * (x * dx + y * dy);
        const auto discriminant = linear * linear - 4 * maximum_distance * (x * x + y * y - .45L * .45L);
        if (discriminant <= 0)
            return;
        const auto lower = std::max(0.L, (-linear - std::sqrt(discriminant)) / (2 * maximum_distance));
        const auto upper = std::min(1.L, (-linear + std::sqrt(discriminant)) / (2 * maximum_distance));
        if (lower >= upper)
            return;
        for (const auto parameter : {lower, upper})
        {
            const auto p = at(parameter);
            examine(normal[0] * (p[1] - .5L) - normal[1] * (p[0] - .5L));
        }
        return;
    }
    if (motion == Motion::ShearReturn)
    {
        // Fit the quadratic from interior values, then check its stationary
        // point and the explicitly one-sided endpoint limits of the same box.
        const auto p25 = normal_velocity(at(.25L)), p50 = normal_velocity(at(.5L)), p75 = normal_velocity(at(.75L));
        const auto quadratic = 8 * (p25 - 2 * p50 + p75);
        const auto linear = 2 * (p75 - p25) - quadratic;
        examine(normal_velocity(at(0)));
        examine(normal_velocity(at(1)));
        if (quadratic != 0)
        {
            const auto critical = -linear / (2 * quadratic);
            if (critical > 0 && critical < 1)
                examine(normal_velocity(at(critical)));
        }
        return;
    }
    // Each translation normal is affine inside a selected cutoff box, even
    // for an oblique native XY chord. Check every actual polygon vertex.
    for (const auto& p : points)
        examine(normal_velocity(p));
}
std::vector<GeneralVoFFaceVolume> rates(const GeneralVoF& source, const Case& c, long double t)
{
    std::vector<GeneralVoFFaceVolume> result;
    SimpleFluid::detail::vof_failure::collective(
        *source.hierarchy()->communicator(),
        [&]
        {
            std::map<AMR::AdaptiveCellKey, XYZ> centers;
            for (const auto& cell : source.support_cells())
                centers.emplace(cell.key, XYZ{cell.centroid[0], cell.centroid[1], cell.centroid[2]});
            for (const auto& tile : source.tiles())
            {
                long double q = 0;
                const auto points = source.finite_face_points<ArithmeticMode::Ordinary>(tile.face_gid);
                std::vector<XYZ> physical;
                for (const auto& p : points)
                    physical.push_back({p.value[0], p.value[1], p.value[2]});
                validate_single_stream_face(std::move(physical), c.motion, mesh_shear(c), t, tile.face_gid);
                if (!tile.exterior)
                {
                    XYZ normal{};
                    for (std::size_t i = 0; i < points.size(); ++i)
                    {
                        XYZ a, b;
                        for (unsigned d = 0; d < 3; ++d)
                        {
                            a[d] = points[i].value[d];
                            b[d] = points[(i + 1) % points.size()].value[d];
                        }
                        normal[0] += a[1] * b[2] - a[2] * b[1];
                        normal[1] += a[2] * b[0] - a[0] * b[2];
                        normal[2] += a[0] * b[1] - a[1] * b[0];
                        a[0] -= mesh_shear(c) * a[1];
                        b[0] -= mesh_shear(c) * b[1];
                        q += edge_integral(a, b, c.motion, t);
                    }
                    long double orientation = 0;
                    for (unsigned d = 0; d < 3; ++d)
                        orientation += normal[d] * (centers.at(*tile.second)[d] - centers.at(tile.first)[d]);
                    if (orientation < 0)
                        q = -q;
                }
                result.push_back({tile.face_gid, double(q)});
            }
        },
        "qualification_face_controls");
    return result;
}
struct Fixture
{
    std::shared_ptr<const GeneralVoF> state;
    std::vector<Cell> cells;
    double maximum_rate = 0;
    std::array<double, 2> original_volume{};
};
Fixture fixture(const Case& c, unsigned n)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto s = mesh_shear(c);
    const unsigned nz = c.motion == Motion::ThreeDimensional ? n : 2;
    if (comm->getRank() == 0)
        std::cout << "Q2_SETUP_BEGIN resolution=" << n
                  << " provider=" << (c.mesh == Mesh::Uniform ? "CartesianAMR" : "ExtrudedAMR") << '\n'
                  << std::flush;
    auto bootstrap = [&]() -> AMR::AdaptiveBootstrap
    {
        if (c.mesh == Mesh::Uniform)
        {
            // Identical Cartesian cells, represented in small immutable root
            // blocks so setup does not repeatedly validate a global XY catalog.
            AMR::AMRGrid grid;
            grid.root_blocks = {n / 2, n / 2, nz / 2};
            grid.block_cells = {2, 2, 2};
            AMR::CartesianAMR provider(grid);
            std::vector<std::uint64_t> owned;
            for (std::uint64_t gid = 0; gid < std::uint64_t(n) * n * nz; ++gid)
            {
                const auto box = provider.cell_box(gid);
                const auto logical_x = (box.lower[0] + box.upper[0]) / 2;
                if (int(logical_x * comm->getSize()) == comm->getRank())
                    owned.push_back(gid);
            }
            return provider.detach_owned(owned, {'q', '2', 'm', 'o', 't', 'i', 'o', 'n'});
        }
        AMR::ExtrudedAMRGrid grid;
        for (unsigned y = 0; y <= n; ++y)
            for (unsigned x = 0; x <= n; ++x)
                grid.xy_nodes.push_back({double((x + s * y) / n), double(y) / n, 0});
        for (unsigned y = 0; y < n; ++y)
            for (unsigned x = 0; x < n; ++x)
            {
                const unsigned p = (n + 1) * y + x;
                grid.base_cells.push_back({p, p + 1, p + n + 2, p + n + 1});
            }
        for (unsigned i = 0; i < n; ++i)
        {
            grid.boundaries.push_back({i, i + 1, "wall"});
            grid.boundaries.push_back({n * (n + 1) + i, n * (n + 1) + i + 1, "wall"});
            grid.boundaries.push_back({i * (n + 1), (i + 1) * (n + 1), "wall"});
            grid.boundaries.push_back({i * (n + 1) + n, (i + 1) * (n + 1) + n, "wall"});
        }
        for (unsigned z = 0; z <= nz; ++z)
            grid.z_edges.push_back(double(z) / nz);
        AMR::ExtrudedAMR provider(grid);
        std::vector<std::uint64_t> owned;
        for (std::uint64_t gid = 0; gid < std::uint64_t(n) * n * nz; ++gid)
        {
            const auto key = provider.cell_key(gid);
            const auto& ancestry = std::get<AMR::ProductCellAncestry>(key.ancestry);
            const auto& root = grid.base_cells.at(ancestry.base.root);
            long double logical_x = 0;
            for (const auto node : root)
                logical_x += grid.xy_nodes[node][0] - s * grid.xy_nodes[node][1];
            logical_x /= root.size();
            if (int(logical_x * comm->getSize()) == comm->getRank())
                owned.push_back(gid);
        }
        return provider.detach_owned(owned, {'q', '2', 'm', 'o', 't', 'i', 'o', 'n'});
    }();
    if (comm->getRank() == 0)
        std::cout << "Q2_SETUP_BOOTSTRAP_READY resolution=" << n << '\n' << std::flush;
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    if (c.mesh == Mesh::CoarseFine)
    {
        std::vector<AMR::AdaptiveMark> marks;
        for (const auto& [key, leaf] : hierarchy->owned())
        {
            const auto p = hierarchy->geometry(leaf)->centroid();
            const auto x = p[0] - double(s) * p[1];
            if (x > .375 && x < .625)
                marks.push_back({key, AMR::AdaptiveSplit::Isotropic});
        }
        hierarchy = hierarchy->apply(hierarchy->plan_adaptation(marks));
    }
    if (comm->getRank() == 0)
        std::cout << "Q2_SETUP_HIERARCHY_READY resolution=" << n << '\n' << std::flush;
    std::vector<AMR::AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    Fixture f;
    const auto polygon = reference_polygon(c.motion, 0);
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        auto ref = cell(hierarchy->geometry(leaf), s);
        order.push_back(key);
        const auto g = phase(ref, c.motion, 0, polygon);
        gas.push_back(double(g));
        carrier.push_back(double(ref.volume() - g));
        f.cells.push_back(std::move(ref));
    }
    GeneralVoFOptions options;
    options.transport_backend = c.backend;
    options.phase_cleanup = {0, {0, 0}};
    double local_volume[2]{};
    for (const auto value : gas)
        local_volume[0] += value;
    for (const auto value : carrier)
        local_volume[1] += value;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_volume, f.original_volume.data());
    const auto* cleanup_switch = std::getenv("SIMPLEFLUID_QUALIFICATION_PHASE_CLEANUP");
    const int enabled = cleanup_switch && std::string_view(cleanup_switch) == "1";
    int minimum_enabled = 0, maximum_enabled = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &enabled, &minimum_enabled);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &enabled, &maximum_enabled);
    SimpleFluid::detail::vof_failure::collective(
        *comm,
        [&]
        {
            if (minimum_enabled != maximum_enabled)
                throw std::invalid_argument("Qualification cleanup switch differs across ranks.");
        },
        "qualification_cleanup_controls");
    if (enabled && c.backend != GeneralVoFTransportBackend::DonorCell)
    {
        options.phase_cleanup.relative_cutoff = 1e-14;
        options.phase_cleanup.maximum_absolute_volume = {1e-13 * f.original_volume[0], 1e-13 * f.original_volume[1]};
    }
    f.state = std::make_shared<const GeneralVoF>(hierarchy, order, gas, carrier, options);
    if (comm->getRank() == 0)
        std::cout << "Q2_SETUP_STATE_READY resolution=" << n << '\n' << std::flush;
    // Compute the Courant scale directly from canonical tile incidence. This
    // includes the boundary cutoff return flow, not just the moving phase core.
    std::map<AMR::AdaptiveCellKey, long double> outgoing;
    const auto flux = rates(*f.state, c, 0);
    for (std::size_t i = 0; i < flux.size(); ++i)
    {
        const auto& tile = f.state->tiles()[i];
        const auto q = flux[i].integrated_volume;
        if (q > 0 && tile.first_owner == comm->getRank())
            outgoing[tile.first] += q;
        if (q < 0 && tile.second && tile.second_owner == comm->getRank())
            outgoing[*tile.second] -= q;
    }
    double local = 0;
    for (std::size_t i = 0; i < order.size(); ++i)
        local = std::max(local, double(outgoing[order[i]] / f.cells[i].volume()));
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &f.maximum_rate);
    return f;
}
std::string cleanup_metrics(const GeneralVoF& source)
{
    const auto& policy = source.options().phase_cleanup;
    const auto& ledger = source.phase_cleanup_state();
    std::ostringstream result;
    result << std::setprecision(std::numeric_limits<long double>::max_digits10)
           << " cleanup_enabled=" << (policy.relative_cutoff > 0)
           << " cleanup_relative_cutoff=" << policy.relative_cutoff << " cleanup_events=" << ledger.cleaned_cells;
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        EXPECT_LE(ledger.absolute_volume[phase], policy.maximum_absolute_volume[phase]);
        EXPECT_LE(std::abs(ledger.signed_volume[phase]), ledger.absolute_volume[phase]);
        result << " cleanup_budget_" << phase << '=' << policy.maximum_absolute_volume[phase]
               << " cleanup_signed_volume_" << phase << '=' << ledger.signed_volume[phase]
               << " cleanup_absolute_volume_" << phase << '=' << ledger.absolute_volume[phase]
               << " cleanup_largest_volume_" << phase << '=' << ledger.largest_absolute_volume[phase];
    }
    return result.str();
}
struct Result
{
    double shape = 0, centroid = 0, width = 1;
    double covariance_xy = 0, phase_change = 0, reference_phase_change = 0;
    std::array<double, 3> center{};
};
std::array<long double, 9> exact_moments(const Case& c, long double time)
{
    auto p = reference_polygon(c.motion, time);
    for (auto& q : p)
        q[0] += mesh_shear(c) * q[1];
    long double twice_area = 0, mx = 0, my = 0, mxx = 0, mxy = 0, myy = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const auto a = p[i], b = p[(i + 1) % p.size()];
        const auto det = a[0] * b[1] - b[0] * a[1];
        twice_area += det;
        mx += (a[0] + b[0]) * det;
        my += (a[1] + b[1]) * det;
        mxx += (a[0] * a[0] + a[0] * b[0] + b[0] * b[0]) * det;
        myy += (a[1] * a[1] + a[1] * b[1] + b[1] * b[1]) * det;
        mxy += (2 * a[0] * a[1] + a[0] * b[1] + b[0] * a[1] + 2 * b[0] * b[1]) * det;
    }
    const auto x = mx / (3 * twice_area), y = my / (3 * twice_area);
    const auto bottom = c.motion == Motion::ThreeDimensional ? .3L + .1L * time : 0.L;
    const auto top = c.motion == Motion::ThreeDimensional ? .55L + .1L * time : 1.L;
    const auto z = (bottom + top) / 2;
    return {x, y, z, mxx / (6 * twice_area), mxy / (12 * twice_area), x * z, myy / (6 * twice_area), y * z,
        (bottom * bottom + bottom * top + top * top) / 3};
}
Result measure(const Fixture& f, const GeneralVoF& state, const Case& campaign, long double t,
    const std::array<double, 3>& initial_center, const std::string& property)
{
    const auto m = campaign.motion;
    const auto polygon = reference_polygon(m, t), initial_polygon = reference_polygon(m, 0);
    // Numerical cell-center moments characterize the piecewise-constant phase
    // distribution. Compare against independent exact polygon/prism moments.
    // The sampled reference is reported separately to expose quadrature bias.
    std::array<double, 27> local{}, global{};
    double minimum_phase = 0, maximum_occupancy = 0;
    double width = 1, global_width = 1;
    for (std::size_t i = 0; i < f.cells.size(); ++i)
    {
        const auto& c = f.cells[i];
        const auto v = c.volume(), ref = phase(c, m, t, polygon);
        const auto g = state.gas_volumes()[i], l = state.carrier_volumes()[i];
        minimum_phase = std::min({minimum_phase, g, l});
        maximum_occupancy = std::max(maximum_occupancy, double(std::abs(g + l - v) / v));
        local[0] += g;
        local[1] += l;
        local[2] += double(ref);
        local[3] += double(v - ref);
        local[4] += double(std::abs(g - ref));
        const auto initial_ref = phase(c, m, 0, initial_polygon);
        local[25] += double(std::abs(g - initial_ref));
        local[26] += double(std::abs(ref - initial_ref));
        for (unsigned d = 0; d < 3; ++d)
        {
            local[5 + d] += g * c.center[d];
            local[8 + d] += ref * c.center[d];
        }
        unsigned k = 0;
        for (unsigned a = 0; a < 3; ++a)
            for (unsigned b = a; b < 3; ++b, ++k)
            {
                local[11 + k] += g * c.center[a] * c.center[b];
                local[17 + k] += ref * c.center[a] * c.center[b];
            }
        if (g > .01L * v && g < .99L * v)
            local[23] += double(v);
        if (ref > .01L * v && ref < .99L * v)
            local[24] += double(v);
        if ((g > 0 && g < v) || (ref > 0 && ref < v))
            width = std::min(width, double(c.width));
    }
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &width, &global_width);
    double global_minimum = 0, global_occupancy = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &minimum_phase, &global_minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &maximum_occupancy, &global_occupancy);
    EXPECT_GE(global_minimum, 0);
    EXPECT_LE(global_occupancy, 1e-13);
    const auto inventory = double(exact_volume(m));
    EXPECT_NEAR(global[2], inventory, 5e-12 * inventory);
    EXPECT_NEAR(global[0], inventory, 2e-11 * inventory);
    EXPECT_NEAR(global[1], 1 - inventory, 2e-11 * (1 - inventory));
    Result result;
    result.shape = global[4] / inventory;
    result.width = global_width;
    const auto exact = exact_moments(campaign, t), initial_exact = exact_moments(campaign, 0);
    double centroid2 = 0, sampled_centroid2 = 0, displacement2 = 0, reference_displacement2 = 0;
    for (unsigned d = 0; d < 3; ++d)
    {
        result.center[d] = global[5 + d] / global[0];
        const auto ref = exact[d];
        centroid2 += std::pow(result.center[d] - ref, 2);
        sampled_centroid2 += std::pow(result.center[d] - global[8 + d] / global[2], 2);
        if (t != 0)
            displacement2 += std::pow(result.center[d] - initial_center[d], 2);
        reference_displacement2 += std::pow(ref - initial_exact[d], 2);
    }
    result.centroid = std::sqrt(centroid2);
    result.covariance_xy = global[12] / global[0] - result.center[0] * result.center[1];
    result.phase_change = global[25] / inventory;
    result.reference_phase_change = global[26] / inventory;
    long double perimeter = 0;
    for (std::size_t i = 0; i < polygon.size(); ++i)
    {
        const auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
        const auto dx = b[0] - a[0] + mesh_shear(campaign) * (b[1] - a[1]);
        perimeter += std::hypot(dx, b[1] - a[1]);
    }
    const auto interface_area = m == Motion::ThreeDimensional ? perimeter * .25L + 2 * area(polygon) : perimeter;
    std::ostringstream report;
    report << std::setprecision(17) << "time=" << double(t) << " shape_error=" << result.shape
           << " cell_center_centroid_error=" << result.centroid
           << " sampled_reference_centroid_error=" << std::sqrt(sampled_centroid2)
           << " interface_width=" << result.width << " cell_center_displacement=" << std::sqrt(displacement2)
           << " reference_displacement=" << std::sqrt(reference_displacement2)
           << " cell_center_covariance_xy=" << result.covariance_xy
           << " phase_change_from_initial=" << result.phase_change
           << " reference_phase_change_from_initial=" << result.reference_phase_change
           << " maximum_relative_occupancy_defect=" << global_occupancy << " gas_inventory=" << global[0]
           << " carrier_inventory=" << global[1] << " thickness_01_99=" << global[23] / interface_area
           << " reference_thickness_01_99=" << global[24] / interface_area << cleanup_metrics(state)
           << " cell_center_raw_moment_errors_vs_exact=";
    for (unsigned k = 0; k < 6; ++k)
        report << (k ? "," : "[") << global[11 + k] / global[0] - exact[3 + k];
    report << "] sampled_reference_raw_moment_errors=";
    for (unsigned k = 0; k < 6; ++k)
        report << (k ? "," : "[") << (global[11 + k] - global[17 + k]) / inventory;
    report << "]";
    testing::Test::RecordProperty(property, report.str());
    if (comm->getRank() == 0)
        std::cout << property << ' ' << report.str() << '\n' << std::flush;
    return result;
}
void write_key(std::ostream& out, const AMR::AdaptiveCellKey& key)
{
    out << key.region << ':' << key.ancestry.index();
    if (const auto* cartesian = std::get_if<AMR::CartesianCellAncestry>(&key.ancestry))
    {
        out << ":level=" << cartesian->level << ":index=";
        for (const auto index : cartesian->index)
            out << index << ',';
    }
    else if (const auto* product = std::get_if<AMR::ProductCellAncestry>(&key.ancestry))
    {
        out << ":base=" << product->base.root << ':' << product->base.root_template << ":path=";
        for (const auto step : product->base.path)
            out << step << ',';
        out << ":z=" << product->axial.root << ":path=";
        for (const auto step : product->axial.path)
            out << step << ',';
    }
}
void dump_failure(const Fixture& f, const Case& c, unsigned n, const std::string& tag, unsigned accepted,
    long double time, long double attempted_dt, const std::string& error)
{
    const auto* destination = std::getenv("SIMPLEFLUID_QUALIFICATION_ARTIFACT_DIR");
    if (!destination)
        return;
    const auto comm = Tpetra::getDefaultComm();
    const auto filename = std::string(name(c.motion)) + "_" + std::to_string(int(c.mesh)) + "_" +
                          std::to_string(int(c.backend)) + "_" + tag + "_N" + std::to_string(n) + "_C" +
                          std::to_string(unsigned(c.courant * 100)) + "_rank" + std::to_string(comm->getRank()) +
                          ".state.txt";
    std::filesystem::create_directories(destination);
    std::ofstream out(std::filesystem::path(destination) / filename);
    if (!out)
        throw std::runtime_error("Could not open qualification failing-state artifact");
    out << std::setprecision(std::numeric_limits<long double>::max_digits10);
    const auto& state = *f.state;
    const auto& options = state.options();
    const auto& reconstruction = options.reconstruction;
    if (const auto exception = std::current_exception())
    {
        SimpleFluid::detail::vof_failure::Record record;
        SimpleFluid::detail::vof_failure::classify(record, exception);
        out << "failure_record presence " << record.present << " category "
            << SimpleFluid::detail::vof_failure::category_name(record.category) << " stage " << record.stage.data()
            << " origin_rank " << record.origin_rank << " cell " << record.cell << " cell_owner " << record.cell_owner
            << " face " << record.face << " donor " << record.donor << " recipient " << record.recipient
            << " arithmetic " << int(record.arithmetic_mode) << " resolution_reason " << record.resolution_reason
            << " geometry_code " << record.geometry_code << " message " << record.message.data()
            << " unmarked_fields_absent\n";
    }
    out << "source_accepted_control " << state.accepted_control() << " hierarchy_partition_generation "
        << state.hierarchy()->partition_generation() << " hierarchy_topology_generation "
        << state.hierarchy()->topology_generation() << '\n';
    out << "error " << error << "\nrank " << comm->getRank() << " size " << comm->getSize() << "\naccepted_steps "
        << accepted << " accepted_time " << time << " attempted_dt " << attempted_dt
        << " required_endpoint 1 configured_backend " << int(c.backend)
        << " actual_family_attempt unpublished expected_family AlgebraicFaceVolume\n"
        << "byte_budget " << options.byte_budget << " stencil_cap " << options.maximum_stencil_cells << " halo_cap "
        << options.maximum_halo_cells << " support_rings " << options.support_rings << " arithmetic "
        << int(options.arithmetic_mode) << " mesh_shear " << mesh_shear(c) << "\nreconstruction measure_only "
        << reconstruction.measure_only << " maximum_offset_iterations " << reconstruction.maximum_offset_iterations
        << " maximum_fit_iterations " << reconstruction.maximum_fit_iterations << " maximum_stencil_cells "
        << reconstruction.maximum_stencil_cells << " maximum_clip_evaluations "
        << reconstruction.maximum_clip_evaluations << " volume_tolerance " << reconstruction.volume_tolerance
        << " fit_tolerance " << reconstruction.fit_tolerance << " maximum_condition "
        << reconstruction.maximum_condition << " orientation_tolerance " << reconstruction.orientation_tolerance
        << " maximum_model_residual " << reconstruction.maximum_model_residual << " minimum_phase_support_fraction "
        << reconstruction.minimum_phase_support_fraction << " minimum_optimized_phase_length_ratio "
        << reconstruction.minimum_optimized_phase_length_ratio << '\n';
    out << cleanup_metrics(state) << "\n";
    const auto cleanup_checkpoint = state.phase_cleanup_checkpoint();
    std::ofstream cleanup_file(std::filesystem::path(destination) / (filename + ".cleanup.bin"), std::ios::binary);
    cleanup_file.write(cleanup_checkpoint.data(), static_cast<std::streamsize>(cleanup_checkpoint.size()));
    std::map<AMR::AdaptiveCellKey, GeneralVoFSupportCell> support;
    for (const auto& sample : state.support_cells())
        support.emplace(sample.key, sample);
    const auto declarations = state.phase_geometry_declarations();
    for (std::size_t i = 0; i < state.owned_order().size(); ++i)
    {
        const auto& key = state.owned_order()[i];
        const auto& sample = support.at(key);
        const auto& roundoff = state.phase_roundoff_bounds()[i];
        const auto& history = state.accumulated_phase_errors()[i];
        const auto& occupancy = state.initial_occupancy_errors()[i];
        out << "owned gid " << sample.cell << " owner " << sample.owner << " key ";
        write_key(out, key);
        out << " gas " << state.gas_volumes()[i] << " liquid " << state.carrier_volumes()[i] << " declaration "
            << int(declarations[i]) << " roundoff " << roundoff.gas << ' ' << roundoff.carrier << " history "
            << history.gas << ' ' << history.carrier << " occupancy " << occupancy.difference << ' '
            << occupancy.admitted_bound << ' ' << occupancy.native_conversion << ' ' << occupancy.phase_input_bound
            << ' ' << occupancy.fit_residual << ' ' << occupancy.fit_bound << ' ' << occupancy.transferred_bound << ' '
            << occupancy.fit_allocation_bound << " stencil_gids";
        for (const auto gid : state.cell_support(sample.cell))
            out << ' ' << gid;
        out << '\n';
    }
    // Joining these rank-local owned inventories by gid reconstructs every
    // retained halo sample. All original finite support geometry is exported.
    for (const auto& [key, sample] : support)
    {
        out << "support gid " << sample.cell << " owner " << sample.owner << " key ";
        write_key(out, key);
        out << " center " << sample.centroid[0] << ' ' << sample.centroid[1] << ' ' << sample.centroid[2] << '\n';
        const auto geometry = sample.finite_geometry;
        const auto certificate =
            PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(geometry)->encode_certificate();
        const auto certificate_name = filename + ".gid" + std::to_string(sample.cell) + ".certificate.bin";
        std::ofstream binary(std::filesystem::path(destination) / certificate_name, std::ios::binary);
        binary.write(certificate.data(), static_cast<std::streamsize>(certificate.size()));
        binary.flush();
        if (!binary)
            throw std::runtime_error("Qualification support certificate write failed");
        out << "certificate " << sample.cell << ' ' << certificate_name << " bytes " << certificate.size() << '\n';
        for (std::size_t p = 0; p < geometry->points().size(); ++p)
        {
            out << "point " << sample.cell << ' ' << p;
            for (const auto value : geometry->points()[p].value)
                out << ' ' << value;
            out << '\n';
        }
        for (std::size_t face = 0; face < geometry->faces().size(); ++face)
        {
            out << "face " << sample.cell << ' ' << face;
            for (const auto vertex : geometry->faces()[face])
                out << ' ' << vertex;
            out << '\n';
        }
    }
    for (const auto& tile : state.tiles())
    {
        out << "tile " << tile.face_gid << " exterior " << tile.exterior << " first " << support.at(tile.first).cell
            << " first_owner " << tile.first_owner;
        if (tile.second)
            out << " second " << support.at(*tile.second).cell << " second_owner " << tile.second_owner;
        out << " nodes";
        for (const auto node : tile.node_gids)
            out << ' ' << node;
        out << '\n';
    }
    out << "fitting_sample_rule independent smaller-phase volume retained; target and neighbor identities above; "
        << "anchor minority to each native finite geometry using the production documented sample contract\n";
    out.flush();
    if (!out)
        throw std::runtime_error("Qualification failing-state artifact write failed");
}
std::optional<Result> run(const Case& c, unsigned n, double courant, const std::string& tag)
{
    const auto comm = Tpetra::getDefaultComm();
    unsigned accepted = 0, limited_steps = 0;
    bool midpoint_reached = false;
    long double time = 0, attempted_dt = 0;
    std::optional<Fixture> working;
    try
    {
        working = fixture(c, n);
        auto& f = *working;
        const auto start = measure(f, *f.state, c, 0, {}, tag + "_initial");
        std::map<std::uint64_t, int> owners;
        for (const auto& tile : f.state->tiles())
            owners.emplace(tile.face_gid, tile.first_owner);
        std::array<double, 16> totals{};
        double maximum_courant = 0;
        EXPECT_GT(f.maximum_rate, 0);
        const long double nominal_dt = courant / f.maximum_rate;
        const auto base_flux = rates(*f.state, c, 0);
        while (time < 1)
        {
            // A known reversal is a physical segment boundary, never hidden in
            // a single net face volume. No retry or smaller dt follows failure.
            const long double segment_end = time < .5L ? .5L : 1.L;
            const auto step_dt = std::min(nominal_dt, segment_end - time);
            attempted_dt = step_dt;
            auto flux = base_flux;
            const auto sign = c.motion == Motion::ShearReturn && time >= .5L ? -1 : 1;
            for (auto& face : flux)
                face.integrated_volume *= sign * double(step_dt);
            const auto trial = f.state->plan_face_transport(flux, double(step_dt), {1, 1000}, accepted + 1);
            const auto prepared = trial.prepared_transport();
            if (!prepared)
                throw std::runtime_error("Face-only planning published no prepared ledger");
            EXPECT_EQ(prepared->backend, c.backend);
            EXPECT_EQ(prepared->family, GeneralVoFTransportFamily::AlgebraicFaceVolume);
            std::array<double, 16> local{}, global{};
            for (const auto& stream : prepared->streams)
                if (owners.at(stream.face) == comm->getRank())
                {
                    const auto theta = stream.theta;
                    EXPECT_GE(theta, 0);
                    EXPECT_LE(theta, 1);
                    const unsigned bin = theta == 0 ? 0 : theta == 1 ? 9 : unsigned(std::floor(8 * theta)) + 1;
                    ++local[bin];
                    if (theta < 1)
                    {
                        ++local[10];
                        local[11] += double(stream.volume);
                        local[12] += double(std::abs(stream.gas_high - stream.gas));
                    }
                    if (c.backend == GeneralVoFTransportBackend::GeometricFace && stream.lower_order)
                    {
                        ++local[13];
                        local[14] += double(stream.volume);
                    }
                }
            local[15] = double(prepared->diagnostics.maximum_outgoing_courant);
            double actual = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local[15], &actual);
            local[15] = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
            maximum_courant = std::max(maximum_courant, actual);
            EXPECT_LE(actual, courant * (1 + 1e-12));
            if (global[10] > 0)
                ++limited_steps;
            for (unsigned i = 0; i < totals.size(); ++i)
                totals[i] += global[i];
            f.state = f.state->apply(trial);
            time += step_dt;
            ++accepted;
            // Check inventories and resolved metrics after every accepted step.
            const auto measured =
                measure(f, *f.state, c, time, start.center, tag + "_step_" + std::to_string(accepted));
            if (time == .5L)
            {
                midpoint_reached = true;
                if (c.motion == Motion::FullRotation)
                {
                    // The exact midpoint centroid moves -.3 in physical x for
                    // every mesh shear. Require half that excursion in the
                    // correct direction for both backend comparisons.
                    EXPECT_GE(start.center[0] - measured.center[0], .15)
                        << "Full-revolution return must include substantial midpoint motion";
                }
                if (c.motion == Motion::ShearReturn)
                {
                    const auto reference_start = exact_moments(c, 0), reference_midpoint = exact_moments(c, .5L);
                    const auto reference_covariance_change =
                        reference_midpoint[4] - reference_midpoint[0] * reference_midpoint[1] -
                        (reference_start[4] - reference_start[0] * reference_start[1]);
                    EXPECT_GT(reference_covariance_change, 0);
                    EXPECT_GE(measured.covariance_xy - start.covariance_xy, .25L * reference_covariance_change)
                        << "Return must include independently resolved positive shear deformation";
                    EXPECT_GT(measured.reference_phase_change, 0);
                    EXPECT_GE(measured.phase_change, .25 * measured.reference_phase_change)
                        << "A stationary phase distribution cannot qualify a complete return cycle";
                    std::ostringstream midpoint_report;
                    midpoint_report << std::setprecision(17)
                                    << "covariance_change=" << measured.covariance_xy - start.covariance_xy
                                    << " reference_covariance_change=" << reference_covariance_change
                                    << " phase_change=" << measured.phase_change
                                    << " reference_phase_change=" << measured.reference_phase_change;
                    testing::Test::RecordProperty(tag + "_midpoint_deformation", midpoint_report.str());
                }
            }
        }
        auto result = measure(f, *f.state, c, 1, start.center, tag + "_final");
        EXPECT_EQ(time, 1);
        EXPECT_TRUE(midpoint_reached) << "Preregistered midpoint checkpoint must be an accepted step boundary";
        if (c.backend == GeneralVoFTransportBackend::GeometricFace)
            EXPECT_EQ(totals[13], 0);
        if (c.motion == Motion::ShearReturn && courant == .8 && !c.temporal &&
            c.backend == GeneralVoFTransportBackend::GeometricFace)
            EXPECT_GE(limited_steps, 4u) << "Sustained limiting gate remains open";
        std::ostringstream report;
        report << std::setprecision(17) << "resolution=" << n << " accepted_steps=" << accepted
               << " endpoint=" << double(time) << " actual_maximum_outgoing_courant=" << maximum_courant
               << " configured_backend="
               << (c.backend == GeneralVoFTransportBackend::GeometricFace ? "GeometricFace" : "DonorCell")
               << " actual_family=AlgebraicFaceVolume ranks=" << comm->getSize() << " limited_steps=" << limited_steps
               << " limited_streams=" << totals[10] << " affected_bulk_volume=" << totals[11]
               << " discarded_gas_correction=" << totals[12] << " fallback_streams=" << totals[13]
               << " fallback_bulk_volume=" << totals[14] << " directed_theta_histogram=[";
        for (unsigned i = 0; i < 10; ++i)
            report << (i ? "," : "") << totals[i];
        report << "]" << cleanup_metrics(*f.state) << " original_input_gas_volume=" << f.original_volume[0]
               << " original_input_carrier_volume=" << f.original_volume[1];
        testing::Test::RecordProperty(tag + "_summary", report.str());
        if (comm->getRank() == 0)
            std::cout << tag << ' ' << report.str() << '\n' << std::flush;
        return result;
    }
    catch (const std::exception& error)
    {
        std::ostringstream report;
        report << "resolution=" << n << " accepted_steps=" << accepted << " accepted_time=" << double(time)
               << " required_endpoint=1 error=" << error.what();
        testing::Test::RecordProperty(tag + "_rejection", report.str());
        ADD_FAILURE() << report.str();
        if (working)
            try
            {
                dump_failure(*working, c, n, tag, accepted, time, attempted_dt, error.what());
            }
            catch (const std::exception& artifact_error)
            {
                ADD_FAILURE() << "Failing-state artifact: " << artifact_error.what();
            }
        return std::nullopt;
    }
}
std::vector<Case> cases()
{
    std::vector<Case> result;
    for (const auto backend : {GeneralVoFTransportBackend::DonorCell, GeneralVoFTransportBackend::GeometricFace})
        for (const auto mesh : {Mesh::Uniform, Mesh::Skew, Mesh::CoarseFine})
            for (const auto motion : {Motion::PlanarPatch, Motion::AxisDrop, Motion::DiagonalDrop, Motion::FullRotation,
                     Motion::ShearReturn, Motion::ThreeDimensional})
            {
                for (const double courant : {.2, .5, .8})
                    result.push_back({motion, mesh, backend, courant});
                result.push_back({motion, mesh, backend, .8, true});
            }
    return result;
}
class AlgebraicMotionQualification : public testing::TestWithParam<Case>
{
};
TEST_P(AlgebraicMotionQualification, CompleteManufacturedHorizon)
{
    if (const auto* enabled = std::getenv("SIMPLEFLUID_RUN_ALGEBRAIC_MOTION"); !enabled || std::string(enabled) != "1")
        GTEST_SKIP() << "Opt-in preregistered campaign: verification/vof/algebraic_motion_manifest.json";
    const auto c = GetParam();
    if (c.temporal)
    {
        std::array<Result, 3> results;
        const double courants[3]{.8, .4, .2};
        for (unsigned i = 0; i < 3; ++i)
        {
            const auto value = run(c, 32, courants[i], "temporal_C" + std::to_string(unsigned(100 * courants[i])));
            ASSERT_TRUE(value.has_value());
            results[i] = *value;
        }
        std::ostringstream report;
        report << std::setprecision(17) << "shape_difference_80_40=" << results[0].shape - results[1].shape
               << " shape_difference_40_20=" << results[1].shape - results[2].shape;
        RecordProperty("temporal_shape_differences", report.str());
        return;
    }
    std::array<Result, 3> results;
    const unsigned resolutions[3]{16, 32, 64};
    for (unsigned i = 0; i < 3; ++i)
    {
        const auto value = run(c, resolutions[i], c.courant, "spatial_N" + std::to_string(resolutions[i]));
        ASSERT_TRUE(value.has_value());
        results[i] = *value;
    }
    const auto order = std::log(results[1].shape / results[2].shape) / std::log(2.);
    RecordProperty("finest_pair_spatial_order", std::to_string(order));
    if (c.backend == GeneralVoFTransportBackend::GeometricFace)
    {
        const double gate = c.motion == Motion::FullRotation ? .05 : c.motion == Motion::ShearReturn ? .1 : .02;
        EXPECT_LE(results[2].shape, gate);
        EXPECT_GE(order, .9);
        if (c.motion != Motion::FullRotation && c.motion != Motion::ShearReturn)
            EXPECT_LE(results[2].centroid, .25 * results[2].width);
    }
}
INSTANTIATE_TEST_SUITE_P(AlgebraicMotion, AlgebraicMotionQualification, testing::ValuesIn(cases()),
    [](const testing::TestParamInfo<Case>& info)
    {
        const auto& c = info.param;
        return std::string(c.backend == GeneralVoFTransportBackend::GeometricFace ? "GeometricFace" : "DonorCell") +
               "_" + name(c.motion) + "_" +
               (c.mesh == Mesh::Uniform   ? "Uniform"
                   : c.mesh == Mesh::Skew ? "Skew"
                                          : "CoarseFine") +
               (c.temporal ? "_Temporal" : "_C" + std::to_string(unsigned(100 * c.courant)));
    });
} // namespace algebraic_motion

// First endpoint check after reconstruction repair. This preserves the
// original finite rectangle, N16, C=.5, midpoint and complete T=1 endpoint.
TEST(GeneralVoFMotionAccuracy, OriginalPlanarPatchN16Endpoint)
{
    if (const auto* enabled = std::getenv("SIMPLEFLUID_RUN_ALGEBRAIC_MOTION"); !enabled || std::string(enabled) != "1")
        GTEST_SKIP() << "Opt-in original prescribed-motion endpoint";
    using namespace algebraic_motion;
    const Case c{Motion::PlanarPatch, Mesh::Uniform, GeneralVoFTransportBackend::GeometricFace, .5};
    const auto result = run(c, 16, .5, "original_planar_patch_N16");
    ASSERT_TRUE(result.has_value());
    RecordProperty("original_endpoint_shape_error", result->shape);
    RecordProperty("original_endpoint_centroid_error", result->centroid);
    // The .02 shape and quarter-cell centroid gates apply to N64 in the
    // original manifest. An N16 endpoint reports accuracy without relabeling
    // those finest-grid gates as passed.
}

// A separately labelled extraction of the preregistered N32 case. It cannot
// replace the failed N16 stress case or establish the three-resolution order.
TEST(GeneralVoFMotionAccuracy, RequiredPlanarPatchN32EndpointDiagnostic)
{
    if (const auto* enabled = std::getenv("SIMPLEFLUID_RUN_ALGEBRAIC_MOTION"); !enabled || std::string(enabled) != "1")
        GTEST_SKIP() << "Opt-in required N32 prescribed-motion endpoint diagnostic";
    using namespace algebraic_motion;
    const Case c{Motion::PlanarPatch, Mesh::Uniform, GeneralVoFTransportBackend::GeometricFace, .5};
    const auto result = run(c, 32, .5, "required_planar_patch_N32");
    ASSERT_TRUE(result.has_value());
    RecordProperty("diagnostic_only", "required N32 case; N16 failure and N64/refinement requirements remain");
    RecordProperty("endpoint_shape_error", result->shape);
    RecordProperty("endpoint_centroid_error", result->centroid);
    // Apply the original finest-case limits here as an additional diagnostic,
    // without claiming that the finest resolution itself was constructed/run.
    EXPECT_LE(result->shape, .02);
    EXPECT_LE(result->centroid, .25 * result->width);
}

TEST(GeneralVoFMotionAccuracy, IndependentAlgebraicPolygonAndMomentOracle)
{
    using namespace algebraic_motion;
    const Case c{Motion::PlanarPatch, Mesh::Skew, GeneralVoFTransportBackend::GeometricFace, .5};
    EXPECT_NEAR(exact_volume(c.motion), .28L * .11L, 1e-17L);
    const auto initial = exact_moments(c, 0), final = exact_moments(c, 1);
    EXPECT_NEAR(initial[0], .35L + .25L * .4L, 1e-17L);
    EXPECT_NEAR(initial[1], .4L, 1e-17L);
    EXPECT_NEAR(final[0] - initial[0], .2L + .25L * .1L, 1e-17L);
    EXPECT_NEAR(final[1] - initial[1], .1L, 1e-17L);
    const Polygon unit{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const Polygon triangle{{0, 0}, {1, 0}, {0, 1}};
    EXPECT_EQ(area(intersection(unit, triangle)), .5L);
    // Analytic flux through a core x-normal square: v_x * .1 * .1.
    const XYZ vertices[4]{{.5L, .4L, .4L}, {.5L, .5L, .4L}, {.5L, .5L, .5L}, {.5L, .4L, .5L}};
    long double flux = 0;
    for (unsigned i = 0; i < 4; ++i)
        flux += edge_integral(vertices[i], vertices[(i + 1) % 4], Motion::DiagonalDrop, 0);
    EXPECT_NEAR(flux, .002L, 1e-18L);
    const Case three{Motion::ThreeDimensional, Mesh::Uniform, GeneralVoFTransportBackend::DonorCell, .5};
    const auto moments = exact_moments(three, 1);
    EXPECT_NEAR(moments[2], .525L, 1e-18L);
    EXPECT_NEAR(moments[8], (.4L * .4L + .4L * .65L + .65L * .65L) / 3, 1e-18L);
}

TEST(GeneralVoFMotionAccuracy, IndependentClosedFluxHasNoHiddenSpatialCounterflow)
{
    using namespace algebraic_motion;
    // Independent derivative of the declared extension, compared against a
    // finite-difference curl of the actual vector-potential implementation.
    const auto analytic = [](XYZ p, Motion motion, long double time)
    {
        const auto fp = [](long double s) { return s < .125L ? 8.L : s > .875L ? -8.L : 0.L; };
        const auto gp = [](long double s) { return s < .125L || s > .875L ? -3.L : 1.L; };
        const auto x = p[0] - .5L, y = p[1] - .5L;
        if (motion == Motion::FullRotation)
        {
            const auto radius2 = x * x + y * y;
            long double factor = -2 * pi;
            if (radius2 >= .45L * .45L)
                factor = 0;
            else if (radius2 > .3L * .3L)
            {
                const auto t = (radius2 - .3L * .3L) / (.45L * .45L - .3L * .3L);
                factor *= (1 - t) * (1 - t) * (1 + 2 * t);
            }
            return XYZ{factor * y, -factor * x, 0};
        }
        if (motion == Motion::ShearReturn)
        {
            const auto sign = time < .5L ? 1 : -1;
            return XYZ{sign * .8L * extension_g(p[1]) * gp(p[1]) * extension_f(p[0]),
                -sign * .4L * extension_g(p[1]) * extension_g(p[1]) * fp(p[0]), 0};
        }
        const auto velocity = translation(motion);
        return XYZ{velocity[0] * extension_f(p[0]) * gp(p[1]) - velocity[1] * extension_g(p[0]) * fp(p[1]) -
                       (motion == Motion::ThreeDimensional ? .1L * extension_g(p[0]) * fp(p[2]) : 0),
            -velocity[0] * fp(p[0]) * extension_g(p[1]) + velocity[1] * gp(p[0]) * extension_f(p[1]),
            motion == Motion::ThreeDimensional ? .1L * gp(p[0]) * extension_f(p[2]) : 0};
    };
    const auto finite_difference = [](XYZ p, Motion motion, long double time)
    {
        constexpr long double h = 1e-7L;
        const auto derivative = [&](unsigned component, unsigned direction)
        {
            auto lo = p, hi = p;
            lo[direction] -= h;
            hi[direction] += h;
            return (potential(hi, motion, time)[component] - potential(lo, motion, time)[component]) / (2 * h);
        };
        return XYZ{derivative(2, 1) - derivative(1, 2), -derivative(2, 0), derivative(1, 0)};
    };
    const auto face_flux = [](unsigned axis, long double fixed, long double a, long double b, long double c,
                               long double d, Motion motion, long double time)
    {
        const unsigned u = (axis + 1) % 3, v = (axis + 2) % 3;
        std::array<XYZ, 4> vertices{};
        for (auto& point : vertices)
            point[axis] = fixed;
        vertices[0][u] = a;
        vertices[0][v] = c;
        vertices[1][u] = b;
        vertices[1][v] = c;
        vertices[2][u] = b;
        vertices[2][v] = d;
        vertices[3][u] = a;
        vertices[3][v] = d;
        long double result = 0;
        for (unsigned edge = 0; edge < 4; ++edge)
            result += edge_integral(vertices[edge], vertices[(edge + 1) % 4], motion, time);
        return result;
    };
    constexpr unsigned n = 16;
    long double maximum_derivative_error = 0, maximum_boundary_flux = 0, maximum_closure = 0;
    unsigned opposing_faces = 0;
    for (const auto motion : {Motion::PlanarPatch, Motion::AxisDrop, Motion::DiagonalDrop, Motion::FullRotation,
             Motion::ShearReturn, Motion::ThreeDimensional})
        for (const auto time : {.25L, .75L})
        {
            if (time == .75L && motion != Motion::ShearReturn)
                continue;
            // All logical N16 faces. Grid lines contain 1/8, 1/2 and 7/8;
            // refinements retain this partition of the sign-constant pieces.
            // The unit-determinant mesh shear preserves these normal flux signs.
            for (unsigned axis = 0; axis < 3; ++axis)
                for (unsigned plane = 0; plane <= n; ++plane)
                    for (unsigned i = 0; i < n; ++i)
                        for (unsigned j = 0; j < n; ++j)
                        {
                            bool positive = false, negative = false;
                            const unsigned u = (axis + 1) % 3, v = (axis + 2) % 3;
                            for (const auto a : {.211324865405187L, .788675134594813L})
                                for (const auto b : {.211324865405187L, .788675134594813L})
                                {
                                    XYZ point{};
                                    point[axis] = static_cast<long double>(plane) / n;
                                    point[u] = (i + a) / n;
                                    point[v] = (j + b) / n;
                                    const auto actual = finite_difference(point, motion, time)[axis];
                                    const auto reference = analytic(point, motion, time)[axis];
                                    maximum_derivative_error =
                                        std::max(maximum_derivative_error, std::abs(actual - reference));
                                    positive = positive || reference > 0;
                                    negative = negative || reference < 0;
                                }
                            if (positive && negative)
                                ++opposing_faces;
                            if (plane == 0 || plane == n)
                                maximum_boundary_flux = std::max(maximum_boundary_flux,
                                    std::abs(face_flux(axis, static_cast<long double>(plane) / n,
                                        static_cast<long double>(i) / n, static_cast<long double>(i + 1) / n,
                                        static_cast<long double>(j) / n, static_cast<long double>(j + 1) / n, motion,
                                        time)));
                        }
            // Cells on both sides of every coordinate transition and midplane,
            // including boundary cells and radial-transition intersections.
            for (const unsigned x : {0u, 1u, 2u, 7u, 8u, 13u, 14u, 15u})
                for (const unsigned y : {0u, 1u, 2u, 7u, 8u, 13u, 14u, 15u})
                    for (const unsigned z : {0u, 1u, 2u, 7u, 8u, 13u, 14u, 15u})
                    {
                        const unsigned index[3]{x, y, z};
                        long double closure = 0;
                        for (unsigned axis = 0; axis < 3; ++axis)
                        {
                            const unsigned u = (axis + 1) % 3, v = (axis + 2) % 3;
                            const auto flux = [&](unsigned plane)
                            {
                                return face_flux(axis, static_cast<long double>(plane) / n,
                                    static_cast<long double>(index[u]) / n, static_cast<long double>(index[u] + 1) / n,
                                    static_cast<long double>(index[v]) / n, static_cast<long double>(index[v] + 1) / n,
                                    motion, time);
                            };
                            closure += flux(index[axis] + 1) - flux(index[axis]);
                        }
                        maximum_closure = std::max(maximum_closure, std::abs(closure));
                    }
        }
    EXPECT_EQ(opposing_faces, 0u);
    EXPECT_LT(maximum_derivative_error, 2e-8L);
    EXPECT_LT(maximum_boundary_flux, 1e-17L);
    EXPECT_LT(maximum_closure, 1e-17L);
    // The former cubic-cutoff counterexample at X=.5,Y in[1/16,1/8]
    // now has constant u=-.6, with no positive normal contribution hidden in q.
    for (const auto y : {.07L, .09L, .12L})
        EXPECT_NEAR(finite_difference({.5L, y, .5L}, Motion::PlanarPatch, 0)[0], -.6L, 1e-10L);
    EXPECT_NEAR(face_flux(0, .5L, 1.L / 16, 1.L / 8, 0, 1, Motion::PlanarPatch, 0), -3.L / 80, 1e-17L);
}

TEST(GeneralVoFMotionAccuracy, ActualNativeFaceWitnessesRejectHiddenSpatialCounterflow)
{
    using namespace algebraic_motion;
    const auto vertical = [](XY a, XY b, long double shear_value)
    {
        std::vector<XYZ> face{{a[0], a[1], .25L}, {b[0], b[1], .25L}, {b[0], b[1], .3125L}, {a[0], a[1], .3125L}};
        for (auto& p : face)
            p[0] += shear_value * p[1];
        return face;
    };
    for (const auto shear_value : {0.L, .25L})
        for (const auto motion : {Motion::PlanarPatch, Motion::AxisDrop, Motion::DiagonalDrop, Motion::FullRotation,
                 Motion::ShearReturn, Motion::ThreeDimensional})
            for (const auto time : {.25L, .75L})
            {
                // Original native Cartesian face and its physical affine shear.
                EXPECT_NO_THROW(validate_single_stream_face(
                    vertical({.3125L, .25L}, {.3125L, .3125L}, shear_value), motion, shear_value, time, 31));
                // A genuinely oblique logical XY chord, still within one box
                // and with one normal sign for every manufactured flow.
                EXPECT_NO_THROW(validate_single_stream_face(
                    vertical({.3125L, .25L}, {.375L, .3125L}, shear_value), motion, shear_value, time, 32));
                std::vector<XYZ> horizontal{
                    {.25L, .25L, .5L}, {.3125L, .25L, .5L}, {.3125L, .3125L, .5L}, {.25L, .3125L, .5L}};
                for (auto& p : horizontal)
                    p[0] += shear_value * p[1];
                EXPECT_NO_THROW(validate_single_stream_face(horizontal, motion, shear_value, time, 33));
            }
    // This chord crosses the radial normal's zero line while entirely inside
    // the rotating region. Its net face flux cannot represent both streams.
    try
    {
        validate_single_stream_face(
            vertical({.625L, .25L}, {.625L, .75L}, .25L), Motion::FullRotation, .25L, .25L, 43981);
        FAIL() << "Opposing radial streams must reject before controls are produced";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(
            std::string(error.what()).find("manufactured fixture requires spatial multistream"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("43981"), std::string::npos);
    }
    EXPECT_THROW(
        validate_single_stream_face(vertical({.5L, .0625L}, {.5L, .1875L}, 0), Motion::PlanarPatch, 0, .25L, 34),
        std::invalid_argument);
}
