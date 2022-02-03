using NearestNeighbors
using StaticArrays
using PyCall
using Graphs
using Meshes
using LinearAlgebra
using NLopt
import Distances.Metric
using QuadGK
using Proj4


const spatial = PyNULL()
const healpy = PyNULL()
function __init__()
    copy!(spatial, pyimport_conda("scipy.spatial", "scipy"))
    copy!(healpy, pyimport_conda("healpy", "healpy"))
end

const Point = SVector{3, Float64}
const LonLat = SVector{2, Float64}
const SphericalCoordinates = SVector{3, Float64}

inner(p1::Point, p2::Point) = p1[1]*p2[1] + p1[2]*p2[2] + p1[3]*p2[3]
radius(p) = sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
cosangle(p::Point, q::Point) = inner(p, q) / (radius(p) * radius(q))
angle(p::Point, q::Point) = acos(cosangle(p, q))


function normalize!(p)
    p /= radius(p)
    return p
end

function normalize(p)
    p / radius(p)
end

abstract type Grid end

struct RegularGrid <: Grid
    parallels::Vector{Float64}
    meridians::Vector{Float64}
    semimajoraxis::Float64
    flattening::Float64
end

struct IrregularGrid <: Grid
    longitude::Vector{Float64}
    latitude::Vector{Float64}
    semimajoraxis::Float64
    flattening::Float64
end

pointcount(grid::IrregularGrid) = length(grid.longitude)
pointcount(grid::RegularGrid) = length(grid.parallels) * length(grid.meridians)


struct EllipsoidalDistance <: Metric
    ellipsoid::Proj4.geod_geodesic
end

function EllipsoidalDistance(a, f)
    g = Proj4.geod_geodesic(a, f)

    EllipsoidalDistance(g)
end

function (dist::EllipsoidalDistance)(x, y)
    Proj4._geod_inverse(dist.ellipsoid, x * 180 / pi, y * 180 / pi)[1]
end

struct ApproximateEllipsoidalDistance <: Metric
    semimajoraxis::Float64
    flattening::Float64
end

function (dist::ApproximateEllipsoidalDistance)(x, y)

    beta1 = geodetic2reduced(x[2], dist.flattening)
    beta2 = geodetic2reduced(y[2], dist.flattening)

    sigma = 2 * asin(0.5 * sqrt((cos(beta2)*cos(y[1]) -  cos(beta1)*cos(x[1]))^2 + (cos(beta2)*sin(y[1]) -  cos(beta1)*sin(x[1]))^2 + (sin(beta2) - sin(beta1))^2))
    if sigma < 1 / dist.semimajoraxis
        return sigma * dist.semimajoraxis
    end

    X = (sigma - sin(sigma)) * sin((beta1 + beta2) * 0.5)^2 * cos((beta2 - beta1) * 0.5)^2 / cos(sigma * 0.5)^2
    Y = (sigma + sin(sigma)) * cos((beta1 + beta2) * 0.5)^2 * sin((beta2 - beta1) * 0.5)^2 / sin(sigma * 0.5)^2

    dist.semimajoraxis * (sigma - 0.5 * dist.flattening * (X + Y))
end

function GeographicGrid(parallel_count, a=Constants.WGS84.semimajor_axis, f=Constants.WGS84.flattening)
    dlat = pi / parallel_count
    meridians = collect(range(-pi + dlat * 0.5, stop=pi - dlat * 0.5, length=2 * parallel_count))
    parallels = collect(range(pi/2 - dlat * 0.5, stop=-pi/2 + dlat * 0.5, length=parallel_count))
    RegularGrid(parallels, meridians, a, f)
end

function ReuterGrid(parallel_count, a=Constants.WGS84.semimajor_axis, f=Constants.WGS84.flattening)
    dlat = pi / parallel_count

    longitude = Vector{Float64}(undef, 1)
    latitude = Vector{Float64}(undef, 1)

    longitude[1] = 0
    latitude[1] = 0.5 * pi
    for beta in range(0.5 * pi - dlat, -0.5 * pi + dlat, parallel_count - 1)

        lat = authalic2geodetic(beta, f)
        meridian_count = floor(Int64, 2 * pi / acos( (cos(dlat) - sin(beta)^2)/cos(beta)^2))
        meridian_longitude = Vector{Float64}(undef, meridian_count)
        meridian_latitude = fill(lat, meridian_count)
        for i in 1:meridian_count
            meridian_longitude[i] = mod((i + 0.5) * 2 * pi / meridian_count + pi, 2*pi) - pi
        end
        sort!(meridian_longitude)
        append!(latitude, meridian_latitude)
        append!(longitude, meridian_longitude)
    end
    append!(longitude, 0.0)
    append!(latitude, -0.5 * pi)
    IrregularGrid(longitude, latitude, a, f)
end

function HEALPix(Nside, a=Constants.WGS84.semimajor_axis, f=Constants.WGS84.flattening)
    nsidelog2 = round(Int, log2(Nside))
    (2^nsidelog2 == Nside) || throw(DomainError(Nside, "Nside must be an integer power of two"))

    npoints = 12 * Nside^2

    longitude = Vector{Float64}(undef, npoints)
    latitude = Vector{Float64}(undef, npoints)

    for k in 1:2*Nside*(Nside - 1)
        i = floor(Int, sqrt(k / 2 - sqrt(floor(k / 2)))) + 1
        j = k - 2*i * (i - 1)
        lon = (Float64(j) - 0.5) * pi / (2i)
        latitude[k], longitude[k] = asin(1 - i^2 / (3*Nside^2)), atan(sin(lon), cos(lon))
    end
    for k in 2*Nside*(Nside - 1)+1:2*Nside * (5*Nside + 1)
        i = floor(Int, (k - 2*Nside*(Nside - 1) - 1) / (4*Nside)) + Nside
        j = Int(mod(k - 2*Nside*(Nside - 1) - 1, 4*Nside)) + 1
        lon = (Float64(j) - 0.5 * (1 + mod(Float64(i + Nside), 2))) * pi / (2*Nside)
        latitude[k], longitude[k] = asin((2*Nside - i) / (1.5 * Nside)),  atan(sin(lon), cos(lon))
    end
    for k in 2*Nside * (5*Nside + 1)+1:npoints
        i = floor(Int, sqrt((npoints - k + 1)/2 - sqrt(floor((npoints - k + 1)/2)))) + 1
        j = Int(4 * i + 1 - (npoints - k + 1 - 2i * (i - 1)))
        lon = (Float64(j) - 0.5) * pi / (2i)
        latitude[k], longitude[k] = asin(-1 + i^2 / (3*Nside^2)), atan(sin(lon), cos(lon))
    end
    IrregularGrid(longitude, authalic2geodetic.(latitude, f), a, f)
end

function subdivideedge(p1, p2, level)
    step_angle = acos(dot(p1, p2)) / (level + 1)
    vec = normalize(cross(cross(p1, p2), p1))

    return [cos(i * step_angle) * p1 + sin(i * step_angle) * vec for i in 1:level]
end

function subdividetriangle(p1, p2, p3, level)

    edge12 = subdivideedge(p1, p2, level)
    edge23 = subdivideedge(p2, p3, level)
    edge31 = subdivideedge(p3, p1, level)

    p = Point[]
    for i in 1:level-1
        for k in 0:i-1
            level - i + k + 1
            e13 = cross(edge12[i + 1], edge31[level - i])
            e12 = cross(edge12[i - k], edge23[level - i + k + 1])
            e23 = cross(edge23[k + 1], edge31[level - k])

            v1 = cross(e13, e12)
            v2 = cross(e23, e13)
            v3 = cross(e23, e12)
            append!(p, [Point(-normalize(normalize(v1) + normalize(v2) + normalize(v3)))])
        end
    end
    return p
end

function HierachicalTriangularMesh(level, a=Constants.WGS84.semimajor_axis, f=Constants.WGS84.flattening)

    vertices = [Point(0, 0, 1), Point(1, 0, 0), Point(0, 1, 0), Point(-1, 0, 0), Point(0, -1, 0), Point(0, 0, -1)]
    triangles = [[1, 5, 2], [2, 5, 3], [3, 5, 4], [4, 5, 1], [1, 0, 4], [4, 0, 3], [3, 0, 2], [2, 0, 1]]
    edges = [[1, 5], [5, 2], [2, 1], [5, 3], [3, 2], [3, 5], [5, 4], [4, 3], [1, 4], [1, 0], [0, 4], [0, 3], [0, 2], [0, 1]]

    for k in 1:length(edges)
        append!(vertices, subdivideedge(vertices[edges[k][1]+1], vertices[edges[k][2]+1], level))
    end

    for k in 1:length(triangles)
       append!(vertices, subdividetriangle(vertices[triangles[k][1]+1], vertices[triangles[k][2]+1], vertices[triangles[k][3]+1], level))
    end

    ll = Matrix{Float64}(undef, length(vertices), 2)
    for k in 1:length(vertices)
        ll[k, 1] = atan(vertices[k][2], vertices[k][1])
        ll[k, 2] = -authalic2geodetic(atan(vertices[k][3], sqrt(vertices[k][1]^2+vertices[k][2]^2)), f)
        # ll[k, 2] = -geocentric2geodetic(pi*0.5-atan(sqrt(vertices[k][1]^2+vertices[k][2]^2),vertices[k][3]), f)
    end
    ll = sortslices(ll[:, end:-1:1], dims=1)[:, end:-1:1]
    IrregularGrid(ll[:, 1], -ll[:, 2], a, f)
end

function GeodesicGrid(level, a=Constants.WGS84.semimajor_axis, f=Constants.WGS84.flattening)

    triangles = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [2, 1, 6], [3, 2, 7], [4, 3, 8],
    [5, 4, 9], [1, 5, 10], [6, 7, 2], [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 6, 1],
    [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]]

    edges = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 1],
    [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 2], [7, 3], [8, 4], [9, 5], [10, 1],
    [6, 7], [7, 8], [8, 9], [9, 10], [10, 6], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10]]

    ratio = pi * 0.5 - acos((cos(72 * pi / 180) + cos(72 * pi / 180) * cos(72 * pi / 180)) / (sin(72 * pi / 180) * sin(72 * pi / 180)))

    lons = [0, 0, 72, 144, 216, 288, 36, 108, 180, 252, 324, 0] * pi / 180
    lats = Vector{Float64}(undef, length(lons))
    lats[1:6].= ratio
    lats[7:end].= -ratio
    lats[1] = 0.5 * pi
    lats[end] = -0.5 * pi

    ll = LonLat.(lons, lats)
    vertices = geodetic2point.(ll, 0, 1, 0)

    for k in 1:length(edges)
        append!(vertices, subdivideedge(vertices[edges[k][1]+1], vertices[edges[k][2]+1], level))
    end

    for k in 1:length(triangles)
       append!(vertices, subdividetriangle(vertices[triangles[k][1]+1], vertices[triangles[k][2]+1], vertices[triangles[k][3]+1], level))
    end

    ll = Matrix{Float64}(undef, length(vertices), 2)
    for k in 1:length(vertices)
        ll[k, 1] = atan(vertices[k][2], vertices[k][1])
        ll[k, 2] = -authalic2geodetic(atan(vertices[k][3], sqrt(vertices[k][1]^2+vertices[k][2]^2)), f)
        # ll[k, 2] = -geocentric2geodetic(pi*0.5-atan(sqrt(vertices[k][1]^2+vertices[k][2]^2),vertices[k][3]), f)
    end
    ll = sortslices(ll[:, end:-1:1], dims=1)[:, end:-1:1]
    IrregularGrid(ll[:, 1], -ll[:, 2], a, f)
end

function SpiralGrid(resolution, a=Constants.WGS84.semimajor_axis, f=Constants.WGS84.flattening)

    function intfun(a, R, c)
        R * sqrt(1 + c^2 * sin(a)^2)
    end

    function optfun(x, grad, sk, R, c)
        I, _ = quadgk((a) -> intfun(a, R, c), 0, x[1])
        abs(sk - I)
    end

    R = authalicradius(a, f)
    c = R * pi / resolution * 2
    S = quadgk((a) -> intfun(a, R, c), 0, pi)[1]
    P = ceil(S / resolution) + 1
    s = S / P
    point_count = Int64(P) + 1

    opt = Opt(:LN_COBYLA, 1)

    thetas = Vector{Float64}(undef, point_count)
    thetas[1] = 0
    for (k, sk) in enumerate(s:s:S-s)
        min_objective!(opt, (x,grad) -> optfun(x,grad,sk,R,c))
        maxeval!(opt, 100)
        ftol_abs!(opt, 0.1)
        _, minx, _ = optimize(opt, [thetas[k]])
        thetas[k+1] = minx[1]
    end
    thetas[end] = pi

    lons = Vector{Float64}(undef, point_count)
    lats = Vector{Float64}(undef, point_count)

    for k in 1:point_count
        lats[k] = authalic2geodetic(pi/2 - thetas[k], f)
        lons[k] = atan(sin(c*thetas[k]), cos(c*thetas[k]))
    end
    IrregularGrid(lons, lats, a, f)
end

function lonlat(grid::Grid)
    ll = Vector{LonLat}(undef, pointcount(grid))
    for k in 1:pointcount(grid)
        ll[k] = LonLat(grid.longitude[k], grid.latitude[k])
    end
    return ll
end

function lonlat(grid::RegularGrid)
    ll = Vector{LonLat}(undef, pointcount(grid))
    for k in 0:length(grid.parallels)-1
        ll[k * length(grid.meridians) + 1:(k + 1) * length(grid.meridians)] = LonLat.(grid.meridians, Ref(grid.parallels[k + 1]))
    end
    return ll
end

function points(grid::Grid)
    ll = lonlat(grid)
    geodetic2point.(ll, 0, grid.semimajoraxis, grid.flattening)
end

function areaweights(grid::Grid)
    fill(4*pi/pointcount(grid), pointcount(grid))
end

function area(ll, a, f)

    ellipsoid = Proj4.geod_geodesic(a, f)
    e2 = ellipsoid.f * (2 - ellipsoid.f)
    e_prime2 = e2 / (1 - e2)
    n = ellipsoid.f / (2 - ellipsoid.f)
    c2 = ellipsoid.a^2 * 0.5 + ellipsoid.b^2 * 0.5 * atanh(sqrt(e2)) / sqrt(e2)

    crossings = 0
    area = 0.0
    for k in 1:length(ll)
        p1 = Vector(ll[k])
        p2 = Vector(ll[(k % length(ll)) + 1])

        _, az1, az2 = Proj4._geod_inverse(ellipsoid, p1 * 180 / pi, p2 * 180 / pi)

        beta1 = geodetic2reduced(p1[2], ellipsoid.f)
        sin_a0 = sind(az1) * cos(beta1)
        epsilon = (sqrt(1 + e_prime2 * (1 - sin_a0*sin_a0)) - 1) / (sqrt(1 - e_prime2 * (1 - sin_a0*sin_a0)) + 1)

        C = [ (2/3 - 4/15*n + 8/105*n^2 + 4/315*n^3 + 16/3465*n^4 + 20/9009*n^5) -
        (1/5 - 16/35*n + 32/105*n^2 - 16/385*n^3 - 64/15015*n^4)*epsilon -
        (2/105 + 32/315*n - 1088/3465*n^2 + 1184/5005*n^3)*epsilon^2 +
        (11/315 - 368/3465*n - 32/6435*n^2)*epsilon^3 +
        (4/1155 + 1088/45045*n)*epsilon^4 + 97/15015*epsilon^5,

        (1/45 - 16/315*n + 32/945*n^2 - 16/3465*n^3 - 64/135135*n^4)*epsilon -
        (2/105 - 64/945*n + 128/1485*n^2 - 1984/45045*n^3)*epsilon^2 -
        (1/105 - 16/2079*n - 5792/135135*n^2)*epsilon^3 +
        (4/1155 - 2944/135135*n)*epsilon^4 + 1/9009*epsilon^5,

        (4/525 - 32/1575*n + 64/3465*n^2 - 32/5005*n^3)*epsilon^2 -
        (8/1575 - 128/5775*n + 256/6825*n^2)*epsilon^3 -
        (8/1925 - 1856/225225*n)*epsilon^4 + 8/10725*epsilon^5,

        (8/2205 - 256/24255*n + 512/45045*n^2)*epsilon^3 -
        (16/8085 - 1024/105105*n)*epsilon^4 - 136/63063*epsilon^5,

        (64/31185 - 512/81081*n)*epsilon^4 - 128/135135*epsilon^5,

        128/99099*epsilon^5]

        sigma1 = atan(sin(geodetic2reduced(p1[2], ellipsoid.f)), cosd(az1) * cos(geodetic2reduced(p1[2], ellipsoid.f)))
        sigma2 = atan(sin(geodetic2reduced(p2[2], ellipsoid.f)), cosd(az2) * cos(geodetic2reduced(p2[2], ellipsoid.f)))

        I12 = 0.0
        for l in 0:length(C)-1
            I12 += C[l+1] * (cos((2*l+1) * sigma2) - cos((2*l+1) * sigma1))
        end
        area += c2 * (az2 - az1) * pi / 180 + ellipsoid.a^2 * e2 * sin_a0 * sqrt(1 - sin_a0*sin_a0) * I12

        dlon = p2[1] - p1[1]
        if p1[1] <= 0 && p2[1] > 0 && atan(sin(dlon), cos(dlon)) > 0
            crossings += 1
        elseif p2[1] <= 0 && p1[1] > 0 && atan(sin(dlon), cos(dlon)) < 0
            crossings -= 1
        end
    end
    area = abs(area)
    if (crossings & 1) > 0
        area -= 2 * pi * c2
    end
    return area
end

function computearea(surfacemesh, a, f)

    areas = Vector{Float64}(undef, length(elements(surfacemesh)))
    for (k, e) in enumerate(Meshes.elements(surfacemesh))
        pts = [point2geodetic(v.coords, a, f)[1] for v in Meshes.vertices(e)]
        areas[k] = abs(area(pts, a, f))
    end
    return areas
end

function areaweights(grid::RegularGrid)

    lonedges = vcat(-pi, grid.meridians[1:end-1] + diff(grid.meridians).*0.5, pi)
    latedges = geodetic2authalic.(vcat(0.5*pi, grid.parallels[1:end-1] + diff(grid.parallels).*0.5, -0.5*pi), Ref(grid.flattening))
    #latedges = vcat(0.5*pi, grid.parallels[1:end-1] + diff(grid.parallels).*0.5, -0.5*pi)

    vec(diff(lonedges).*transpose(2*sin.(abs.(diff(latedges)*0.5)).*cos.(grid.parallels)))
end

function toirregular(grid::RegularGrid)
    ll = lonlat(grid)
    lons = Vector{Float64}(undef, length(ll))
    lats = Vector{Float64}(undef, length(ll))
    for k in 1:length(ll)
        lons[k] = ll[k][1]
        lats[k] = ll[k][2]
    end
    IrregularGrid(lons, lats, grid.semimajoraxis, grid.flattening)
end

function nn(grid, points)
    # metric = EllipsoidalDistance(grid.semimajoraxis, grid.flattening)
    metric = ApproximateEllipsoidalDistance(grid.semimajoraxis, grid.flattening)
    tree = NearestNeighbors.BallTree(lonlat(grid), metric, reorder=false)

    NearestNeighbors.nn(tree, points)
end

function graph(grid)
    pts = points(grid)
    py = spatial.ConvexHull(pts)

    simplices = convert(Vector{Vector{Int}}, py."simplices")
    g = Graphs.SimpleGraph(pointcount(grid))
    for k in 1:length(simplices)
        for l in 0:length(simplices[k])-1
            Graphs.add_edge!(g, simplices[k][l + 1] + 1, simplices[k][(l +1) % length(simplices[k]) + 1] + 1)
        end
    end
    return g
end

function mesh(grid::Grid)
    pts = points(grid)
    py = spatial.ConvexHull(pts)

    simplices = convert(Vector{Vector{Int}}, py."simplices")
    tuples = Vector{Tuple}(undef, length(simplices))
    for k in 1:length(simplices)
        x = simplices[k]
        if det(hcat(pts[x[1]+1], pts[x[2]+1], pts[x[3]+1])) < 0
            tuples[k] = Tuple([x[3]+1, x[2]+1, x[1]+1])
        else
            tuples[k] = Tuple([x[1]+1, x[2]+1, x[3]+1])
        end
    end
    connec = Meshes.connect.(tuples, Meshes.Ngon)
    Meshes.SimpleMesh(Meshes.Point.(pts), connec)
end

# function mesh(grid::RegularGrid)
#     pts = points(grid)

#     tuples = Vector{Tuple}(undef, length(pts))
#     M = length(grid.meridians)
#     for m in 1:M
#         tuples[m] = (m>1 ? m-1 : M, m<M ? m+1 : 1, M + m)
#     end
#     for p in 2:length(grid.parallels)-1
#         for m in 1:M
#             tuples[(p-1)*M+m] = (m>1 ? m-1 : M, m<M ? m+1 : 1, M + m, m-M).+(p-1)*M
#         end
#     end
#     for m in 1:M
#         tuples[(length(grid.parallels)-1)*M+m] = (m>1 ? m-1 : M, m<M ? m+1 : 1, m - M).+(length(grid.parallels)-1)*M
#     end
#     connec = Meshes.connect.(tuples, Meshes.Ngon)
#     Meshes.SimpleMesh(Meshes.Point.(pts), connec)
# end

function surfaceelements(grid::Grid)

    metric = ApproximateEllipsoidalDistance(grid.semimajoraxis, grid.flattening)
    opt = Opt(:LN_COBYLA, 2)

    function optfun(x, grad, p1, p2, p3, metric)

        s1m = metric(x, p1)
        s2m = metric(x, p2)
        s3m = metric(x, p3)

        return (s1m - s2m)^2 + (s1m - s3m)^2
    end

    pts = points(grid)
    py = spatial.ConvexHull(pts)

    simplices = convert(Vector{Vector{Int}}, py."simplices")
    regions = Vector{Vector{Int64}}(undef, length(pts))
    centroids = Vector{Point}(undef, length(simplices))
    for k in 1:length(simplices)

        for i in simplices[k]
            try
                append!(regions[i + 1], k)
            catch
                regions[i + 1] = [k]
            end
        end
        p = sum(pts[[i+1 for i in simplices[k]]]) / length(simplices[k])
        x0, _ = point2geodetic(p, grid.semimajoraxis, grid.flattening)
        p1, _ = point2geodetic(pts[simplices[k][1] + 1], grid.semimajoraxis, grid.flattening)
        p2, _ = point2geodetic(pts[simplices[k][2] + 1], grid.semimajoraxis, grid.flattening)
        p3, _ = point2geodetic(pts[simplices[k][3] + 1], grid.semimajoraxis, grid.flattening)

        min_objective!(opt, (x,grad) -> optfun(x,grad,p1,p2,p3,metric))
        maxeval!(opt, 100)
        ftol_abs!(opt, 1)
        _, minx, _ = optimize(opt, x0)
        centroids[k] = geodetic2point(minx, 0, grid.semimajoraxis, grid.flattening)
    end
    tuples = Vector{Tuple}(undef, length(regions))
    for k in 1:length(regions)
        idx = regions[k]
        center = pts[k]

        e = cross(center, [0, 0, 1])
        normalize!(e)
        n = cross(e, center)

        tuples[k] = Tuple([idx[i] for i in sortperm(atan.([dot(c, e) for c in centroids[idx]], [dot(c, n) for c in centroids[idx]]))])
    end

    connec = Meshes.connect.(tuples, Meshes.Ngon)
    Meshes.SimpleMesh([Meshes.Point(p[1], p[2], p[3]) for p in centroids], connec)
end

function surfaceelements(grid::RegularGrid)

    lonedges = vcat(-pi, grid.meridians[1:end-1] + diff(grid.meridians).*0.5)
    latedges = grid.parallels[1:end-1] + diff(grid.parallels).*0.5
    P = length(grid.parallels)
    M = length(grid.meridians)
    vertices = Vector{Meshes.Point3}(undef, 2 + (P-1) * M)
    vertices[1] = Meshes.Point(geodetic2point(LonLat(0, 0.5*pi), 0, grid.semimajoraxis, grid.flattening))
    for k in 0:P-2
        ll = LonLat.(lonedges, Ref(latedges[k+1]))
        vertices[k * M + 2:(k + 1) * M+1] = geodetic2point.(ll, 0, grid.semimajoraxis, grid.flattening)
    end
    vertices[end] = Meshes.Point(geodetic2point(LonLat(0, -0.5*pi), 0, grid.semimajoraxis, grid.flattening))

    tuples = Vector{Tuple}(undef, pointcount(grid))
    for m in 1:M
        tuples[m] = (1, m+1, m<M ? m+2 : 2)
    end
    for p in 2:length(grid.parallels)-1
        for m in 1:M
            tuples[(p-1)*M+m] = (m+1, m+1+M, (m<M ? m+2 : 2)+M, m<M ? m+2 : 2).+(p-2)*M
        end
    end
    for m in 1:M
        tuples[(length(grid.parallels)-1)*M+m] = (m+1+(P-2)*M, length(vertices), ((m<M ? m+2 : 2)+(P-2)*M))
    end
    connec = Meshes.connect.(tuples, Meshes.Ngon)
    Meshes.SimpleMesh(vertices, connec)
end

function geodetic2point(lonlat, height::Number, a, f)

    e2 = f * (2 - f)
    N = a / sqrt(1 - e2 * sin(lonlat[2])^2)

    x = (N + height) * cos(lonlat[2]) * cos(lonlat[1])
    y = (N + height) * cos(lonlat[2]) * sin(lonlat[1])
    z = ((1 - e2)*N + height) * sin(lonlat[2])

    Point(x, y, z)
end

function point2spherical(point)
    r = radius(point)
    colat = atan(sqrt(point[1]^2 + point[2]^2), point[3])
    lon = atan(point[2], point[1])
    return r, colat, lon
end

function point2geodetic(point, a, f, max_iter=10, threshold=1e-6)

    if iszero(f)
        r, colat, lon = point2spherical(point)
        return LonLat(lon, pi * 0.5 - colat), r - a
    end

    p2 = point[1]^2 + point[2]^2
    e2 = f * (2 - f)
    k = 1 / (1 - e2)

    lon = atan(point[2], point[1])
    lat = atan(k * point[3], sqrt(p2))
    h = 0

    for _ in 1:max_iter
        c = (p2 + (1 - e2) * point[3]^2 * k^2)^1.5 / (a * e2)
        k = 1 + (p2 + (1 - e2) * point[3]^2 * k^3) / (c - p2)

        h1 = (1 / k - (1 - e2)) * sqrt(p2 + point[3]^2 * k^2) / e2
        lat1 = atan(k * point[3], sqrt(p2))

        if max(abs(h - h1), abs(lat - lat1) * a) < threshold
            break
        end
        h = h1
        lat = lat1
    end

    return LonLat(lon, lat), h
end

function authalic2geodetic(beta, f)
    if iszero(f)
        return beta
    end

    e2 = f * (2 - f)

    return beta +
    (1/3 * e2 + 31/180 * e2^2 + 517/5040 * e2^3 + 120389/181400 * e2^4 + 1362254/29937600 * e2^5) * sin(2 * beta) +
    (23/360 * e2^2 + 251/3780 * e2^3 + 102287/1814400 * e2^4 + 450739/997920 * e2^5) * sin(4 * beta) +
    (761/45360 * e2^3 + 47561/1814400 * e2^4 + 434501/14968800 * e2^5) * sin(6 * beta) +
    (6059/1209600 * e2^4 + 625511/59875200 * e2^5) * sin(8 * beta) +
    (48017/29937600 * e2^5) * sin(10 * beta)
end

function authalicradius(a, f)
    if iszero(f)
        return a
    end

    e = sqrt(f * (2 - f))
    b = a * (1 - f)

    return sqrt((a^2 + b^2 / e * log((a/b)*(1+e)))*0.5)
end

function geodetic2authalic(latitude, f)
    if iszero(f) || iszero(abs(latitude) - pi/2)
        return latitude
    end

    e = sqrt(f * (2 - f))
    q = (1 - e^2) * sin(latitude) / (1 - (e*sin(latitude))^2) - (1 - e^2) / (2 * e) * log((1 - e * sin(latitude)) / (1 + e * sin(latitude)))
    q0 = (1 - e^2) / (1 - e^2) - (1 - e^2) / (2 * e) * log((1 - e) / (1 + e))

    asin(q / q0)
end

function geodetic2geocentric(latitude, f)
    if iszero(f)
        return latitude
    end

    atan((1 - f)^2 * sin(latitude), cos(latitude))
end

function geocentric2geodetic(beta, f)
    if iszero(f)
        return beta
    end

    atan(sin(beta), (1 - f)^2 * cos(beta))
end

function geodetic2reduced(latitude, f)
    if iszero(f)
        return latitude
    end

    atan((1 - f) * sin(latitude), cos(latitude))
end

function reduced2geodetic(beta, f)
    if iszero(f)
        return beta
    end

    atan(sin(beta), (1 - f) * cos(beta))
end
