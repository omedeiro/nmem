from tdgl import Polygon, Device, Layer
from tdgl.geometry import box
import qnngds.geometry as qg
import numpy as np
from nmem.simulation.pytdgl.sim.constants import XI, D, LONDONL, SIGMA, length_units

def make_device():
    layer = Layer(coherence_length=XI, london_lambda=LONDONL, thickness=D, conductivity=SIGMA, gamma=23.8)
    mem = qg.memory_v4()
    p = mem.polygons[0].polygons[0]
    pout = np.vstack((p[:149], p[-2:], p[:1]))
    pin = np.vstack((p[150:-2], p[150:151]))
    film = Polygon("film", points=pout)
    hole = Polygon("center", points=pin).buffer(0)
    source = Polygon("source", points=box(2.1, 0.1)).translate(dx=0.5, dy=3.4)
    drain = Polygon("drain", points=box(2.1, 0.1)).translate(dx=0.5, dy=-3.4)
    return Device("weak_link", layer=layer, film=film, holes=[hole], terminals=[source, drain], probe_points=[(0.5, 3), (0.5, -3)], length_units=length_units)


