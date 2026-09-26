"""Ground-plane geometry: homography from 4 or more points (a 2D/2.5D ground-plane model, not a 3D twin) and zone tests."""
from __future__ import annotations

import math

Point = tuple[float, float]
Matrix = list[list[float]]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("degenerate point correspondences (collinear points?)")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def homography_from_points(img_pts: list[Point], ground_pts: list[Point]) -> Matrix:
    """Exact homography from 4 image->ground correspondences (image px -> metres)."""
    if len(img_pts) != 4 or len(ground_pts) != 4:
        raise ValueError("need exactly 4 correspondences")
    a, b = [], []
    for (x, y), (u, v) in zip(img_pts, ground_pts):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = _solve(a, b)
    return [h[0:3], h[3:6], [h[6], h[7], 1.0]]


def _normaliser(pts: list[Point]) -> Matrix:
    """Hartley normalisation: centroid to the origin, mean distance sqrt(2) (keeps the least squares well posed)."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    d = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts) or 1.0
    s = math.sqrt(2) / d
    return [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def homography_lsq(img_pts: list[Point], ground_pts: list[Point]) -> Matrix:
    """Least-squares homography from >= 4 correspondences (normalised DLT with h33 = 1, normal equations)."""
    if len(img_pts) < 4 or len(img_pts) != len(ground_pts):
        raise ValueError("need at least 4 matching correspondences")
    if len(img_pts) == 4:
        return homography_from_points(img_pts, ground_pts)
    ti, tg = _normaliser(img_pts), _normaliser(ground_pts)
    ip = [apply_h(ti, p) for p in img_pts]
    gp = [apply_h(tg, p) for p in ground_pts]
    rows, rhs = [], []
    for (x, y), (u, v) in zip(ip, gp):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(8)] for i in range(8)]
    atb = [sum(r[i] * b for r, b in zip(rows, rhs)) for i in range(8)]
    h = _solve(ata, atb)
    hn = [h[0:3], h[3:6], [h[6], h[7], 1.0]]
    out = _matmul(invert_h(tg), _matmul(hn, ti))
    k = out[2][2] if abs(out[2][2]) > 1e-12 else 1.0
    return [[v / k for v in row] for row in out]


def fit_homography(img_pts: list[Point], ground_pts: list[Point], robust_threshold_m: float = 0.5,
                   max_subsets: int = 2000) -> tuple[Matrix, dict]:
    """Image->ground homography from 4 or more correspondences, with an honest error estimate.

    4 points: the exact solve (no redundancy, so no internal error estimate). 5+ points: RANSAC over 4-point
    subsets (exhaustive when there are at most `max_subsets`, else a fixed-seed sample): the subset whose homography
    puts the most points within `robust_threshold_m` of their surveyed ground position wins, then all its inliers
    are refitted by least squares. Points outside are reported as outliers (a mistyped or mis-clicked point).
    With 5+ inliers the leave-one-out RMSE (refit without each inlier, measure that inlier) is reported: an
    estimate of the ground error at points the fit has not seen.
    """
    n = len(img_pts)
    if n != len(ground_pts) or n < 4:
        raise ValueError("need at least 4 matching image/ground correspondences")
    if n == 4:
        return homography_from_points(img_pts, ground_pts), {"fit": "exact (4 points)", "points": 4, "inliers": 4,
                                                             "outliers": [], "fit_rms_m": 0.0, "loo_rmse_m": None}
    import itertools
    import random
    combos = list(itertools.combinations(range(n), 4)) if math.comb(n, 4) <= max_subsets else None
    if combos is None:
        rng = random.Random(0)
        combos = [tuple(sorted(rng.sample(range(n), 4))) for _ in range(max_subsets)]
    best, best_key = None, None
    for c in combos:
        try:
            h = homography_from_points([img_pts[i] for i in c], [ground_pts[i] for i in c])
            errs = [math.dist(apply_h(h, img_pts[i]), ground_pts[i]) for i in range(n)]
        except ValueError:  # degenerate subset (collinear) or a point beyond the horizon
            continue
        inl = [i for i, e in enumerate(errs) if e <= robust_threshold_m]
        key = (len(inl), -sum(min(e, robust_threshold_m) ** 2 for e in errs))
        if best_key is None or key > best_key:
            best, best_key = inl, key
    if not best or len(best) < 4:
        raise ValueError("no 4 calibration points agree on one ground plane (check the point pairs)")
    ii = [img_pts[i] for i in best]
    gg = [ground_pts[i] for i in best]
    h = homography_lsq(ii, gg)
    fit_rms = rms_error(h, ii, gg)
    loo = None
    if len(best) >= 5:
        sq = []
        for k in range(len(best)):
            hk = homography_lsq(ii[:k] + ii[k + 1:], gg[:k] + gg[k + 1:])
            sq.append(math.dist(apply_h(hk, ii[k]), gg[k]) ** 2)
        loo = math.sqrt(sum(sq) / len(sq))
    info = {"fit": "RANSAC + least squares", "points": n, "inliers": len(best),
            "outliers": [i for i in range(n) if i not in best], "robust_threshold_m": robust_threshold_m,
            "fit_rms_m": round(fit_rms, 3), "loo_rmse_m": round(loo, 3) if loo is not None else None}
    return h, info


def apply_h(h: Matrix, p: Point) -> Point:
    x, y = p
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-12:
        raise ValueError("point maps to infinity (above horizon?)")
    return ((h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w)


def invert_h(h: Matrix) -> Matrix:
    (a, b, c), (d, e, f), (g, i, j) = h
    det = a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)
    inv = [[e * j - f * i, c * i - b * j, b * f - c * e],
           [f * g - d * j, a * j - c * g, c * d - a * f],
           [d * i - e * g, b * g - a * i, a * e - b * d]]
    return [[v / det for v in row] for row in inv]


def rms_error(h: Matrix, img_pts: list[Point], ground_pts: list[Point]) -> float:
    """RMS ground error (metres) on held-out check points."""
    if not img_pts:
        return 0.0
    errs = [math.dist(apply_h(h, p), g) ** 2 for p, g in zip(img_pts, ground_pts)]
    return math.sqrt(sum(errs) / len(errs))


def point_in_polygon(p: Point, poly: list[Point]) -> bool:
    x, y = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _seg_dist(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def signed_depth(p: Point, poly: list[Point]) -> float:
    """Distance to the zone boundary: positive inside, negative outside (metres)."""
    d = min(_seg_dist(p, poly[i - 1], poly[i]) for i in range(len(poly)))
    return d if point_in_polygon(p, poly) else -d


def foot_point(bbox: tuple[float, float, float, float]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)
