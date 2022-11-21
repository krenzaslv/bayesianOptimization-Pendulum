import math

def clamp(t, a):
    return abs(t) / t * a if abs(t) > a else t

def angleDiff(a, b):
    c = b - a
    return (c + math.pi) % (2 * math.pi) - math.pi
