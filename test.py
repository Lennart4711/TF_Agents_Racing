import random
import time

points = [random.randint(-1000, 1000) for i in range(1000)]


def get_collision_point(a, b):
    x1 = a[0][0]
    x2 = a[1][0]
    y1 = a[0][1]
    y2 = a[1][1]
    x3 = b[0][0]
    x4 = b[1][0]
    y3 = b[0][1]
    y4 = b[1][1]

    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    # Line segments are parallel
    if denominator == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    # Do the lines intersect in the given segments?
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        intersectionX = x1 + (ua * (x2 - x1))
        intersectionY = y1 + (ua * (y2 - y1))
        return [intersectionX, intersectionY]
    else:
        return None


# Time get_collision_point for 1000 iterations on random points in a and b
end = 0
for _ in range(1000000):
    a, b, c, d = [random.sample(points, 2) for _ in range(4)]
    start = time.perf_counter()
    get_collision_point([a, b], [c, d])
    end += time.perf_counter() - start
print(f"get_collision_point: {end}")
