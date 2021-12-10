using GeographicLib
using NearestNeighbors
using Distances
using StaticArrays
using PyCall
using Graphs
using Meshes
using LinearAlgebra
using NLopt


const spatial = PyNULL()

function __init__()
    copy!(spatial, pyimport_conda("scipy.spatial", "scipy"))
end

const Point = SVector{3, Float64}
const LonLat = SVector{2, Float64}

norm(p) = sqrt(p[1]^2 + p[2]^2 + p[3]^2)

function normalize!(p)
    n = norm(p)
    p /= n
    return n
end

function normalize(p)
    p / norm(p)
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
    ellipsoid::GeographicLib.Geodesic
end

function EllipsoidalDistance(a, f)
    g = Geodesic(a, f)

    EllipsoidalDistance(g)
end

function (dist::EllipsoidalDistance)(x, y)
    GeographicLib.Inverse(dist.ellipsoid, x[2] * 180 / pi, x[1] * 180 / pi, y[2] * 180 / pi, y[1] * 180 / pi, GeographicLib.Geodesics.DISTANCE).s12
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

function GeographicGrid(parallel_count, a=6378137.0, f=1/298.257223563)
    dlat = pi / parallel_count
    meridians = collect(range(-pi + dlat * 0.5, stop=pi - dlat * 0.5, length=2 * parallel_count))
    parallels = collect(range(pi/2 - dlat * 0.5, stop=-pi/2 + dlat * 0.5, length=parallel_count))
    RegularGrid(parallels, meridians, a, f)
end

function ReuterGrid(parallel_count, a=6378137.0, f=1/298.257223563)
    dlat = pi / parallel_count

    longitude = Vector{Float64}(undef, 1)
    latitude = Vector{Float64}(undef, 1)

    longitude[1] = 0
    latitude[1] = 0.5 * pi

    for k in 2:parallel_count
        theta = (k - 1) * dlat

        lat = authalic2geodetic(0.5 * pi - theta, f)
        meridian_count = floor(Int64, 2 * pi / acos( (cos(dlat) - cos(theta)^2)/sin(theta)^2))
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

function GeodesicGrid(level, a=6378137.0, f=1/298.257223563)

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

    for k in 1:length(edges)
        append!(vertices, subdivideedge(vertices[edges[k][1]+1], vertices[edges[k][2]+1], level))
    end

    for k in 1:length(triangles)
       append!(vertices, subdividetriangle(vertices[triangles[k][1]+1], vertices[triangles[k][2]+1], vertices[triangles[k][3]+1], level))
    end

    ll = Matrix{Float64}(undef, length(vertices), 2)
    for k in 1:length(vertices)
        ll[k, 1] = atan(vertices[k][2], vertices[k][1])
        ll[k, 2] = -authalic2geodetic(pi*0.5-atan(sqrt(vertices[k][1]^2+vertices[k][2]^2),vertices[k][3]), f)
        # ll[k, 2] = -geocentric2geodetic(pi*0.5-atan(sqrt(vertices[k][1]^2+vertices[k][2]^2),vertices[k][3]), f)
    end
    ll = sortslices(ll[:, end:-1:1], dims=1)[:, end:-1:1]
    IrregularGrid(ll[:, 1], -ll[:, 2], a, f)
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

function mesh(grid)
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

function voronoi(grid)

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

function geodetic2point(lonlat, height::Number, a, f)

    e2 = f * (2 - f)
    N = a / sqrt(1 - e2 * sin(lonlat[2])^2)

    x = (N + height) * cos(lonlat[2]) * cos(lonlat[1])
    y = (N + height) * cos(lonlat[2]) * sin(lonlat[1])
    z = ((1 - e2)*N + height) * sin(lonlat[2])

    Point(x, y, z)
end

function point2spherical(point)
    r = norm(point)
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

function geodetic2authalic(latitude, f)
    if iszero(f)
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
