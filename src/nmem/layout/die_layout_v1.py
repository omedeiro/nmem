# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:51:12 2022

@author: omedeiro
"""

from __future__ import absolute_import, division, print_function

# import colang as mc
import string

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
    D = pg.gridsweep(
        om.die_cell,
        param_y={"text1": ["NMEM " + sub for sub in list(string.ascii_uppercase[0:7])]},
        param_x={"text2": list(string.digits[1:8])},
        spacing=(0, 0),
    )
    E = Device("nMem")
    loop_size = np.tile(np.linspace(0.5, 3.5, 7), 7)
    for i, s in zip(D.references, loop_size):
        F = Device("nMem_cell")
        nMem = F << qg.memory_v4(right_extension=s)
        nMem.movex(nMem.ports[1].midpoint[0], i.ports[1].midpoint[0])
        nMem.movey(nMem.ports[3].midpoint[1], i.ports[4].midpoint[1])
        F << pr.route_smooth(nMem.ports[1], i.ports[1], layer=1)
        F << pr.route_smooth(nMem.ports[2], i.ports[2], layer=1)
        F << pr.route_smooth(nMem.ports[3], i.ports[4], layer=2)
        F << pr.route_smooth(nMem.ports[4], i.ports[3], layer=2)

        spacer = F << pg.rectangle(size=(300, 600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E << F
    # D.flatten()
    D << E
    D.name = "nMemTestDie"
    return D


def nMemTestDieV1(loop_size=1, text1="A", text2="1"):
    F = Device("nMem_cell")
    pad = F << om.die_cell(text1=text1, text2=text2)
    nMem = F << qg.memory_v4(right_extension=loop_size)
    nMem.movex(nMem.ports[1].midpoint[0], pad.ports[1].midpoint[0])
    nMem.movey(nMem.ports[3].midpoint[1], pad.ports[4].midpoint[1])
    F << pr.route_smooth(nMem.ports[1], pad.ports[1], layer=1)
    F << pr.route_smooth(nMem.ports[2], pad.ports[2], layer=1)
    F << pr.route_smooth(nMem.ports[3], pad.ports[4], layer=2)
    F << pr.route_smooth(nMem.ports[4], pad.ports[3], layer=2)
    F.flatten()

    return F


# D = nMemTestDieV1()
# text1 = list(string.ascii_uppercase[0:7])
# text2 = list(string.digits[1:8])
# loop_size = np.tile(np.linspace(0.5,3.5,7), 7)
# D_list=[]
# for txt1 in text1:
#     for s, txt2 in zip(loop_size, text2):
#         D_list.append(nMemTestDieV1(s, "NMEM "+txt1, txt2))
# D = pg.grid(D_list, shape=(7,7))
# qp(D)


def testStructureDie():
    D = pg.gridsweep(
        om.die_cell,
        param_y={"text1": ["TEST " + sub for sub in list(string.ascii_uppercase[0:7])]},
        param_x={"text2": list(string.digits[1:8])},
        spacing=(0, 0),
    )
    E = Device("test_structures")
    wire_width = np.tile(np.linspace(0.1, 0.7, 7), 7)
    for i, s in zip(D.references, wire_width):
        F = Device("test_structure_cell")

        lithosteps = F << pg.litho_steps(
            line_widths=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
            line_spacing=2,
            height=50,
            layer=0,
        )
        lithosteps.rotate(-90)
        lithosteps.move(lithosteps.center, i.center + (-200, -200))
        lithostar = F << pg.litho_star(line_width=1, diameter=100, layer=0)
        lithostar.move(lithostar.center, i.center + (-120, -120))

        lithosteps = F << pg.litho_steps(
            line_widths=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
            line_spacing=2,
            height=50,
            layer=1,
        )
        lithosteps.rotate(-90)
        lithosteps.move(lithosteps.center, i.center + (-200, 200))
        lithostar = F << pg.litho_star(line_width=1, diameter=100, layer=1)
        lithostar.move(lithostar.center, i.center + (-120, 120))

        lithosteps = F << pg.litho_steps(
            line_widths=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
            line_spacing=2,
            height=50,
            layer=2,
        )
        lithosteps.rotate(-90)
        lithosteps.move(lithosteps.center, i.center + (200, 200))
        lithostar = F << pg.litho_star(line_width=1, diameter=100, layer=2)
        lithostar.move(lithostar.center, i.center + (120, 120))

        lithocal = F << pg.litho_calipers(layer1=0, layer2=1)
        lithocal.move(lithocal.center, i.center + (140, -200))
        lithocal = F << pg.litho_calipers(layer1=2, layer2=1)
        lithocal.rotate(180)
        lithocal.move(lithocal.center, i.center + (140, -220))

        lithocal = F << pg.litho_calipers(layer1=0, layer2=1)
        lithocal.rotate(90)
        lithocal.move(lithocal.center, i.center + (200, -140))
        lithocal = F << pg.litho_calipers(layer1=2, layer2=1)
        lithocal.rotate(-90)
        lithocal.move(lithocal.center, i.center + (220, -140))

        line = F << pr.route_smooth(i.ports[1], i.ports[2], width=s, layer=1)

        spacer = F << pg.rectangle(size=(500, 600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E << F
    D.flatten()
    D << E
    D.name = "testStructureDie"
    return D


# qp(testStructureDie())


def nTronTestDie():
    D = pg.gridsweep(
        om.die_cell,
        param_y={
            "text1": ["NTRON " + sub for sub in list(string.ascii_uppercase[0:7])]
        },
        param_x={"text2": list(string.digits[1:8])},
        param_defaults={"ports_ground": ["S"]},
        spacing=(0, 0),
    )
    E = Device("nTron")
    choke_offset = np.tile(np.linspace(0, 3, 7), 7)
    for i, s in zip(D.references, choke_offset):
        F = Device("nTron_cell")
        nTron = F << qg.ntron_v2(choke_offset=s)
        nTron.movex(nTron.ports[1].midpoint[0], i.ports[1].midpoint[0])
        nTron.movey(nTron.ports[3].midpoint[1], i.ports[3].midpoint[1])

        F << pr.route_smooth(nTron.ports[1], i.ports[1], layer=1)
        F << pr.route_smooth(nTron.ports[2], i.ports[2], layer=1)
        F << pr.route_smooth(nTron.ports[3], i.ports[4], layer=1)
        F << pr.route_smooth(nTron.ports[4], i.ports[3], layer=1)

        spacer = F << pg.rectangle(size=(500, 600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E << F
    D.flatten()
    D << E
    D.name = "nTronTestDie"
    return D


def snspdTestDie():
    D = pg.gridsweep(
        om.die_cell,
        param_y={"text1": ["SPD " + sub for sub in list(string.ascii_uppercase[0:7])]},
        param_x={"text2": list(string.digits[1:8])},
        param_defaults={"ports_ground": ["S"]},
        spacing=(0, 0),
    )
    E = Device("SNSPD")
    snspd_area = np.tile(np.linspace(10, 250, 7), 7)
    for i, s in zip(D.references, snspd_area):
        F = Device("snspd_cell")
        spd = F << pg.snspd_expanded(
            wire_width=0.1, wire_pitch=0.4, size=(s, s), layer=1
        )
        spd.rotate(-90)
        spd.move(spd.center, i.center)

        F << pr.route_smooth(spd.ports[1], i.ports[1], layer=1)
        F << pr.route_smooth(spd.ports[2], i.ports[2], layer=1)

        spacer = F << pg.rectangle(size=(500, 600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E << F
    D.flatten()
    D << E
    D.name = "snspdTestDie"
    return D


def nMemArray():
    D = pg.gridsweep(
        om.die_cell_v2,
        param_y={
            "text1": ["NMEMs " + sub for sub in list(string.ascii_uppercase[0:3])]
        },
        param_x={"text2": list(string.digits[1:4])},
        param_defaults={
            "ports": {"N": 8, "E": 8, "W": 8, "S": 8},
            "ports_ground": ["E", "S"],
        },
        spacing=(0, 0),
    )
    E = Device("nMem Array")
    array_size = np.tile([2, 4, 8], 4)
    for i, s in zip(D.references, array_size):
        F = Device("nmem_array_cell")
        nmem = F << om.memory_array(s, s)
        nmem.move(nmem.center, i.center)

        if s == 2:
            F << pr.route_smooth(
                nmem.ports[1],
                i.ports[4],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[2],
                i.ports[5],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[3],
                i.ports[12],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[4],
                i.ports[13],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[5],
                i.ports[28],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[6],
                i.ports[29],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

            F << pr.route_smooth(
                nmem.ports[7],
                i.ports[20],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[8],
                i.ports[21],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

        if s == 4:
            F << pr.route_smooth(
                nmem.ports[1],
                i.ports[3],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[2],
                i.ports[4],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[3],
                i.ports[5],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[4],
                i.ports[6],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[5],
                i.ports[11],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[6],
                i.ports[12],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[7],
                i.ports[13],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[8],
                i.ports[14],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[9],
                i.ports[27],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[10],
                i.ports[28],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[11],
                i.ports[29],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[12],
                i.ports[30],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

            F << pr.route_smooth(
                nmem.ports[13],
                i.ports[19],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[14],
                i.ports[20],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[15],
                i.ports[21],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[16],
                i.ports[22],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

        if s == 6:
            F << pr.route_smooth(
                nmem.ports[1],
                i.ports[2],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[2],
                i.ports[3],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[3],
                i.ports[4],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[4],
                i.ports[5],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[5],
                i.ports[6],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[6],
                i.ports[7],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[7],
                i.ports[10],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[8],
                i.ports[11],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[9],
                i.ports[12],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[10],
                i.ports[13],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[11],
                i.ports[14],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[12],
                i.ports[15],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[13],
                i.ports[26],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[14],
                i.ports[27],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[15],
                i.ports[28],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[16],
                i.ports[29],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[17],
                i.ports[30],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[18],
                i.ports[31],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

            F << pr.route_smooth(
                nmem.ports[19],
                i.ports[18],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[20],
                i.ports[19],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[21],
                i.ports[20],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[22],
                i.ports[21],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[23],
                i.ports[22],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[24],
                i.ports[23],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

        if s == 8:
            F << pr.route_smooth(
                nmem.ports[1],
                i.ports[1],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[2],
                i.ports[2],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[3],
                i.ports[3],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[4],
                i.ports[4],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[5],
                i.ports[5],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[6],
                i.ports[6],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[7],
                i.ports[7],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[8],
                i.ports[8],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[9],
                i.ports[9],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[10],
                i.ports[10],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[11],
                i.ports[11],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[12],
                i.ports[12],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[13],
                i.ports[13],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[14],
                i.ports[14],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[15],
                i.ports[15],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                nmem.ports[16],
                i.ports[16],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )

            F << pr.route_smooth(
                nmem.ports[17],
                i.ports[25],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[18],
                i.ports[26],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[19],
                i.ports[27],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[20],
                i.ports[28],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[21],
                i.ports[29],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[22],
                i.ports[30],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[23],
                i.ports[31],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[24],
                i.ports[32],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )

            F << pr.route_smooth(
                nmem.ports[25],
                i.ports[17],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[26],
                i.ports[18],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[27],
                i.ports[19],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[28],
                i.ports[20],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[29],
                i.ports[21],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[30],
                i.ports[22],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[31],
                i.ports[23],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                nmem.ports[32],
                i.ports[24],
                path_type="Z",
                length1=50,
                length2=20,
                layer=2,
            )

        spacer = F << pg.rectangle(size=(500, 600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E << F
    D.flatten()
    D << E
    D.name = "nMemArrayDie"
    return D


def snspdArray():
    D = pg.gridsweep(
        om.die_cell_v2,
        param_y={"text1": ["SPDs " + sub for sub in list(string.ascii_uppercase[0:3])]},
        param_x={"text2": list(string.digits[1:4])},
        param_defaults={
            "ports": {"N": 8, "E": 8, "W": 8, "S": 8},
            "ports_ground": ["E", "S"],
        },
        spacing=(0, 0),
    )
    E = Device("SNSPDArray")
    array_size = np.tile([2, 4, 8], 4)
    for i, s in zip(D.references, array_size):
        F = Device("snspd_array")
        spd = F << om.single_bit_array(s, s)
        spd.move(spd.center, i.center)

        if s == 2:
            F << pr.route_smooth(
                spd.ports[1], i.ports[3], path_type="Z", length1=20, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[2], i.ports[6], path_type="Z", length1=20, length2=50, layer=1
            )

            F << pr.route_smooth(
                spd.ports[3],
                i.ports[11],
                path_type="Z",
                length1=20,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[4],
                i.ports[14],
                path_type="Z",
                length1=20,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                spd.ports[5].rotate(180),
                i.ports[28],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[6].rotate(180),
                i.ports[29],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

            F << pr.route_smooth(
                spd.ports[7].rotate(180),
                i.ports[20],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[8].rotate(180),
                i.ports[21],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

        if s == 4:
            F << pr.route_smooth(
                spd.ports[1], i.ports[3], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[2], i.ports[4], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[3], i.ports[5], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[4], i.ports[6], path_type="Z", length1=50, length2=50, layer=1
            )

            F << pr.route_smooth(
                spd.ports[5],
                i.ports[11],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[6],
                i.ports[12],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[7],
                i.ports[13],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[8],
                i.ports[14],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )

            F << pr.route_smooth(
                spd.ports[9].rotate(180),
                i.ports[27],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[10].rotate(180),
                i.ports[28],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[11].rotate(180),
                i.ports[29],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[12].rotate(180),
                i.ports[30],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

            F << pr.route_smooth(
                spd.ports[13].rotate(180),
                i.ports[19],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[14].rotate(180),
                i.ports[20],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[15].rotate(180),
                i.ports[21],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[16].rotate(180),
                i.ports[22],
                path_type="Z",
                length1=50,
                length2=50,
                layer=2,
            )

        # if s==6:
        #     F<<pr.route_smooth(spd.ports[1], i.ports[2], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[2], i.ports[3], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[3], i.ports[4], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[4], i.ports[5], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[5], i.ports[6], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[6], i.ports[7], path_type='Z', length1=50, length2=50, layer=1)

        #     F<<pr.route_smooth(spd.ports[7], i.ports[10], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[8], i.ports[11], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[9], i.ports[12], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[10], i.ports[13], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[11], i.ports[14], path_type='Z', length1=50, length2=50, layer=1)
        #     F<<pr.route_smooth(spd.ports[12], i.ports[15], path_type='Z', length1=50, length2=50, layer=1)

        #     F<<pr.route_smooth(spd.ports[13], i.ports[26], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[14], i.ports[27], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[15], i.ports[28], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[16], i.ports[29], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[17], i.ports[30], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[18], i.ports[31], path_type='Z', length1=50, length2=50, layer=2)

        #     F<<pr.route_smooth(spd.ports[19], i.ports[18], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[20], i.ports[19], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[21], i.ports[20], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[22], i.ports[21], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[23], i.ports[22], path_type='Z', length1=50, length2=50, layer=2)
        #     F<<pr.route_smooth(spd.ports[24], i.ports[23], path_type='Z', length1=50, length2=50, layer=2)

        if s == 8:
            F << pr.route_smooth(
                spd.ports[1], i.ports[1], path_type="Z", length1=50, length2=20, layer=1
            )
            F << pr.route_smooth(
                spd.ports[2], i.ports[2], path_type="Z", length1=50, length2=20, layer=1
            )
            F << pr.route_smooth(
                spd.ports[3], i.ports[3], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[4], i.ports[4], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[5], i.ports[5], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[6], i.ports[6], path_type="Z", length1=50, length2=50, layer=1
            )
            F << pr.route_smooth(
                spd.ports[7], i.ports[7], path_type="Z", length1=50, length2=20, layer=1
            )
            F << pr.route_smooth(
                spd.ports[8], i.ports[8], path_type="Z", length1=50, length2=20, layer=1
            )

            F << pr.route_smooth(
                spd.ports[9], i.ports[9], path_type="Z", length1=50, length2=20, layer=1
            )
            F << pr.route_smooth(
                spd.ports[10],
                i.ports[10],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[11],
                i.ports[11],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[12],
                i.ports[12],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[13],
                i.ports[13],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[14],
                i.ports[14],
                path_type="Z",
                length1=50,
                length2=50,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[15],
                i.ports[15],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )
            F << pr.route_smooth(
                spd.ports[16],
                i.ports[16],
                path_type="Z",
                length1=50,
                length2=20,
                layer=1,
            )

            F << pr.route_smooth(
                spd.ports[17].rotate(180),
                i.ports[25],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[18].rotate(180),
                i.ports[26],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[19].rotate(180),
                i.ports[27],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[20].rotate(180),
                i.ports[28],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[21].rotate(180),
                i.ports[29],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[22].rotate(180),
                i.ports[30],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[23].rotate(180),
                i.ports[31],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[24].rotate(180),
                i.ports[32],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )

            F << pr.route_smooth(
                spd.ports[25].rotate(180),
                i.ports[17],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[26].rotate(180),
                i.ports[18],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[27].rotate(180),
                i.ports[19],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[28].rotate(180),
                i.ports[20],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[29].rotate(180),
                i.ports[21],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[30].rotate(180),
                i.ports[22],
                path_type="Z",
                length1=20,
                length2=50,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[31].rotate(180),
                i.ports[23],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )
            F << pr.route_smooth(
                spd.ports[32].rotate(180),
                i.ports[24],
                path_type="Z",
                length1=20,
                length2=20,
                layer=2,
            )

        spacer = F << pg.rectangle(size=(500, 600), layer=3)
        spacer.move(spacer.center, i.center)

        F.flatten()
        E << F
    D.flatten()
    D << E
    D.name = "snspdArrayDie"
    return D


# %%
A = Device("wafer")

A << pg.ring(radius=76e3 / 2, layer=99)

Ndie = 21

die = pg.basic_die(size=(10e3, 10e3), die_name="")
die_list = np.tile(die, 25)

die_array = pg.grid(die_list, shape=(5, 5), spacing=(0, 0))

# die_array.remove(die_array.references[0])
# die_array.remove(die_array.references[3])
# die_array.remove(die_array.references[18])
# die_array.remove(die_array.references[-1])

die_array.move(die_array.center, (0, 0))
DA = A << die_array


for i in [1, 9, 15, 23]:
    a = A << nMemTestDie()
    a.move(a.center, die_array.references[i].center)

for i in [2, 10, 12, 14, 22]:
    a = A << testStructureDie()
    a.move(a.center, die_array.references[i].center)

for i in [3, 5, 19, 21]:
    a = A << nTronTestDie()
    a.move(a.center, die_array.references[i].center)

for i in [6, 8, 16, 18]:
    a = A << snspdTestDie()
    a.move(a.center, die_array.references[i].center)

for i in [7, 17]:
    a = A << snspdArray()
    a.move(a.center, die_array.references[i].center)

for i in [11, 13]:
    a = A << nMemArray()
    a.move(a.center, die_array.references[i].center)

qp(A)

# %%

# fieldnumber = 0
# B = Device()
# for a in A.references:
#     # print(a)
#     loc1 = a.origin

#     die = a.parent
#     for d in die.references:
#         loc2 = d.origin

#         field = d.parent
#         for f in field.references:
#             fieldnumber=fieldnumber+1
#             fieldID = "F"+str("%s" % fieldnumber).zfill(4)
#             loc3 = f.origin
#             D = Device();
#             subfield = D<<f.parent
#             subfield.move(destination=loc1+loc2+loc3)
#             filepath = r'G:\My Drive\___Projects\_electronics\wafer_scale_process\GDS_CELLS'
#             filename = filepath+"\\"+fieldID+"_"+subfield.parent.name+".gds"
#             # D.write_gds(filename)
#             B<<D
# # A<<field
# qp(B)
