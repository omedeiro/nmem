# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:39:38 2023

@author: omedeiro
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import phidl.geometry as pg
import phidl.routing as pr
import qnngds.geometry as qg
import qnngds.omedeiro_v3 as om
from phidl import Device, set_quickplot_options
from phidl import quickplot as qp

set_quickplot_options(show_ports=True, show_subports=True)

# %%


def nMemTestDie():
    D = om.die_cell()
    F = Device("nMem")
    nMem = F << qg.memory_v4(right_extension=1)
    nMem.movex(nMem.ports[1].midpoint[0], D.ports[1].midpoint[0])
    nMem.movey(nMem.ports[3].midpoint[1], D.ports[4].midpoint[1])
    bridge = pg.straight(size=(10, 180), layer=4)
    B1 = F << bridge
    B1.connect(B1.ports[1], D.ports[3])
    B2 = F << bridge
    B2.connect(B2.ports[1], D.ports[4])
    F << pr.route_smooth(nMem.ports[1], D.ports[1], layer=1)
    F << pr.route_smooth(nMem.ports[2], D.ports[2], layer=1)
    F << pr.route_smooth(
        nMem.ports[3], B2.ports[2], layer=2, width=(nMem.ports[3].width, 2)
    )
    F << pr.route_smooth(
        nMem.ports[4], B1.ports[2], layer=2, width=(nMem.ports[4].width, 2)
    )
    B1.movex(-20)
    B2.movex(20)
    F << D

    spacer = F << pg.rectangle(size=(200, 600), layer=3)
    spacer.move(spacer.center, D.center)
    F.flatten()
    F.name = "nMemTestDie"

    qp(F)
    return F


def nMemArray_n2():
    D = om.die_cell_v3(ports={"N": 2, "E": 2, "W": 2, "S": 2}, ports_ground=["E", "S"])
    F = Device("nmem_array_cell")
    nmem = F << om.memory_array(2, 2)
    nmem.move(nmem.center, D.center)

    layers = np.append(np.tile(1, 4), np.tile(2, 4))
    print(int(len(nmem.ports) / 2))
    for i in range(1, int(len(nmem.ports) / 2) + 1):
        F << pr.route_smooth(
            nmem.ports[i], D.ports[i], path_type="Z", length1=20, length2=50, layer=1
        )

    for i in range(int(len(nmem.ports) / 2) + 1, len(nmem.ports) + 1):
        B1 = F << pg.straight(size=(20, 120), layer=4)
        B1.connect(B1.ports[1], D.ports[i])
        B2 = F << pg.straight(size=(20, 30), layer=2)
        B2.connect(B2.ports[1], B1.ports[2].rotate(180))
        F << pr.route_sharp(
            nmem.ports[i], B1.ports[2], path_type="Z", length1=20, length2=50, layer=2
        )

    spacer = F << pg.rectangle(size=(400, 400), layer=3)
    spacer.move(spacer.center, D.center)
    F << D
    return F


def port_norm(port):
    return np.linalg.norm(port.midpoint)


def port_nearest(a_portlist, b_portlist):
    b_portlist2 = b_portlist
    new_list = []
    for pa in a_portlist:
        distance = []
        for pb in b_portlist2:
            distance.append(np.linalg.norm(pa.midpoint - pb.midpoint))
        idx = np.argmin(distance)
        new_list.append(b_portlist2[idx])
        del b_portlist2[idx]
    return new_list


def nMemArray_n4(n=4):
    D = om.die_cell_v3(
        ports={"N": n, "E": n, "W": n, "S": n},
        ports_ground=["E", "S"],
        size2=(500, 500),
    )
    F = Device("nmem_array_cell")
    nmem = F << om.memory_array(n, n)
    nmem.move(nmem.center, D.center)

    layers = np.append(np.tile(1, 2 * n), np.tile(2, 2 * n))

    for i in range(1, 2 * n + 1):
        F << pr.route_smooth(
            nmem.ports[i], D.ports[i], path_type="Z", length1=50, length2=50, layer=1
        )

    heaterInt = F << pg.compass_multi((300, 300), ports={"W": n, "E": n})
    heaterInt_ports = list(heaterInt.ports.values())
    heaterInt_ports.sort(key=port_norm, reverse=True)
    nmem_portlist = list(nmem.ports.values())
    nmem_portlist2 = nmem_portlist[2 * n : 4 * n]
    nmem_portlist2 = port_nearest(heaterInt_ports, nmem_portlist2)

    for i in range(0, 2 * n):
        F << pr.route_sharp(
            nmem_portlist2[i],
            heaterInt_ports[i].rotate(180),
            length1=10,
            length2=30,
            width=(nmem_portlist2[i].width, 10),
            path_type="Z",
            layer=2,
        )
        extension = F << pg.straight(size=(10, 20), layer=2)
        extension.connect(extension.ports[1], heaterInt_ports[i].rotate(180))
    padPorts = list(D.ports.values())
    padPorts = padPorts[2 * n : 4 * n]
    padPorts.sort(key=port_norm, reverse=True)
    heaterInt_ports = port_nearest(padPorts, heaterInt_ports)
    for i in range(0, 2 * n):
        F << pr.route_sharp(
            padPorts[i],
            heaterInt_ports[i],
            length1=30,
            length2=30,
            width=padPorts[i].width,
            path_type="Z",
            layer=4,
        )

    F.remove(heaterInt)

    spacer = F << pg.rectangle(size=(400, 600), layer=3)
    spacer.move(spacer.center, D.center)
    F << D
    return F


def nMemArray_n8(n=8):
    D = om.die_cell_v3(
        ports={"N": n, "E": n, "W": n, "S": n},
        ports_ground=["E", "S"],
        size=(250 * 2 * n, 250 * 2 * n),
        size2=(600, 600),
    )
    F = Device("nmem_array_cell")
    nmem = F << om.memory_array(n, n)
    nmem.move(nmem.center, D.center)

    layers = np.append(np.tile(1, 2 * n), np.tile(2, 2 * n))

    for i in range(1, 2 * n + 1):
        F << pr.route_smooth(
            nmem.ports[i],
            D.ports[i],
            path_type="Z",
            length1=50,
            length2=20,
            layer=1,
            radius=1,
        )

    heaterIntsizeX = abs(D.ports[1].midpoint[0] - D.ports[2 * n].midpoint[0]) - 80
    heaterIntsizeY = abs(D.ports[2 * n].midpoint[1]) + 100
    heaterIntsize = np.array([heaterIntsizeX, heaterIntsizeY])
    heaterInt = F << pg.compass_multi(heaterIntsize, ports={"W": n, "E": n})
    heaterInt_ports = list(heaterInt.ports.values())
    heaterInt_ports.sort(key=port_norm, reverse=True)

    nmem_portlist = list(nmem.ports.values())
    nmem_portlist2 = nmem_portlist[2 * n : 4 * n]
    nmem_portlist2 = port_nearest(heaterInt_ports, nmem_portlist2)

    for i in range(0, 2 * n):
        F << pr.route_sharp(
            nmem_portlist2[i],
            heaterInt_ports[i].rotate(180),
            length1=10,
            length2=30,
            width=(nmem_portlist2[i].width, 10),
            path_type="Z",
            layer=2,
        )
        extension = F << pg.straight(size=(10, 20), layer=2)
        extension.connect(extension.ports[1], heaterInt_ports[i].rotate(180))
    padPorts = list(D.ports.values())
    padPorts = padPorts[2 * n : 4 * n]
    padPorts.sort(key=port_norm, reverse=True)
    heaterInt_ports = port_nearest(padPorts, heaterInt_ports)
    for i in range(0, 2 * n):
        F << pr.route_sharp(
            padPorts[i],
            heaterInt_ports[i],
            length1=30,
            length2=30,
            width=padPorts[i].width,
            path_type="Z",
            layer=4,
        )

    F.remove(heaterInt)
    spacerLoc1 = D.ports[1].midpoint + np.array([-20, 10])
    spacerLoc2 = D.ports[2 * n].midpoint + np.array([20, -10])
    spacer = F << pg.rectangle(size=spacerLoc1 - spacerLoc2, layer=3)
    spacer.move(spacer.center, D.center)
    F << D
    qp(F)
    return F


nMemArray_n8()


def snspdArray(n=8):
    D = om.die_cell_v3(
        ports={"N": n, "E": n, "W": n, "S": n},
        ports_ground=["E", "S"],
        size=(250 * 2 * n, 250 * 2 * n),
        size2=(650, 650),
        ground_width=300,
    )

    F = Device("snspd_array")
    spd = F << om.single_bit_array(n, n)
    spd.move(spd.center, D.center)

    for i in range(1, 2 * n + 1):
        F << pr.route_smooth(
            spd.ports[i],
            D.ports[i],
            path_type="Z",
            length1=50,
            length2=20,
            layer=1,
            radius=1,
        )

    heaterIntsizeX = abs(D.ports[1].midpoint[0] - D.ports[2 * n].midpoint[0]) - 80
    heaterIntsizeY = abs(D.ports[2 * n].midpoint[1]) + 100
    heaterIntsize = np.array([heaterIntsizeX, heaterIntsizeY])
    heaterInt = F << pg.compass_multi(heaterIntsize, ports={"W": n, "E": n})
    heaterInt_ports = list(heaterInt.ports.values())
    heaterInt_ports.sort(key=port_norm, reverse=True)

    spd_portlist = list(spd.ports.values())
    spd_portlist2 = spd_portlist[2 * n : 4 * n]
    spd_portlist2 = port_nearest(heaterInt_ports, spd_portlist2)

    for i in range(0, 2 * n):
        F << pr.route_sharp(
            spd_portlist2[i].rotate(180),
            heaterInt_ports[i].rotate(180),
            length1=10,
            length2=30,
            width=(spd_portlist2[i].width, 10),
            path_type="Z",
            layer=2,
        )
        extension = F << pg.straight(size=(10, 20), layer=2)
        extension.connect(extension.ports[1], heaterInt_ports[i].rotate(180))
    padPorts = list(D.ports.values())
    padPorts = padPorts[2 * n : 4 * n]
    padPorts.sort(key=port_norm, reverse=True)
    heaterInt_ports = port_nearest(padPorts, heaterInt_ports)
    for i in range(0, 2 * n):
        F << pr.route_sharp(
            padPorts[i],
            heaterInt_ports[i],
            length1=30,
            length2=30,
            width=padPorts[i].width,
            path_type="Z",
            layer=4,
        )

    F.remove(heaterInt)

    spacerLoc1 = D.ports[1].midpoint + np.array([-20, 10])
    spacerLoc2 = D.ports[2 * n].midpoint + np.array([20, -10])
    spacer = F << pg.rectangle(size=spacerLoc1 - spacerLoc2, layer=3)
    spacer.move(spacer.center, D.center)

    # F.flatten()
    # E<<F
    # D.flatten()
    F << D
    D.name = "snspdArrayDie"
    qp(F)
    return D
