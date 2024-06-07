# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:51:12 2022

@author: omedeiro
"""


from __future__ import division, print_function, absolute_import
import numpy as np
from phidl import Device, CrossSection, Path
import phidl.geometry as pg
import phidl.routing as pr
from phidl import quickplot as qp
# import colang as mc
import string
from datetime import datetime
import os
import sys
from time import sleep
import time
import phidl.path as ppz

from phidl.device_layout import _parse_layer, DeviceReference

from argparse import Namespace    



sys.path.append(r'Q:\qnnpy')
sys.path.append(r'Q:\qnngds')
import qnngds.omedeiro_v3 as om
import qnnpy.functions.functions as qf
import qnngds.utilities as qu
import qnngds.geometry as qg
from phidl import set_quickplot_options
set_quickplot_options(show_ports=True, show_subports=True)


def port_norm(port):
    return np.linalg.norm(port.midpoint)

def port_nearest(a_portlist, b_portlist):
    b_portlist2 = b_portlist
    new_list = []
    for pa in a_portlist:
        distance=[]
        for pb in b_portlist2:    
            distance.append(np.linalg.norm(pa.midpoint-pb.midpoint))
        idx = np.argmin(distance)
        new_list.append(b_portlist2[idx])
        del b_portlist2[idx]
    return new_list


def testStructureDie():
    D = pg.gridsweep(om.die_cell, param_y={"text1": ['TEST ' + sub for sub in list(string.ascii_uppercase[0:7])]}, param_x={'text2':list(string.digits[1:8])}, spacing=(0,0))
    E = Device('test_structures')
    wire_width = np.tile(np.linspace(0.1, 0.7, 7), 7)
    for i, s in zip(D.references, wire_width):
        F = Device('test_structure_cell')

        # lithosteps = F<<pg.litho_steps(line_widths = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], line_spacing=2, height=50, layer=0) 
        # lithosteps.rotate(-90)
        # lithosteps.move(lithosteps.center, i.center+(-200, -200))
        # lithostar = F<<pg.litho_star(line_width=1, diameter=100, layer=0)
        # lithostar.move(lithostar.center, i.center+(-120, -120))
        

        lithosteps = F<<pg.litho_steps(line_widths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1], line_spacing=2, height=50, layer=1) 
        lithosteps.rotate(-90)
        lithosteps.move(lithosteps.center, i.center+(-200, 200))
        lithostar = F<<pg.litho_star(line_width=1, diameter=100, layer=1)
        lithostar.move(lithostar.center, i.center+(-120, 120))
        
        # lithosteps = F<<pg.litho_steps(line_widths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1], line_spacing=2, height=50, layer=2) 
        # lithosteps.rotate(-90)
        # lithosteps.move(lithosteps.center, i.center+(200, 200))
        # lithostar = F<<pg.litho_star(line_width=1, diameter=100, layer=2)
        # lithostar.move(lithostar.center, i.center+(120, 120))
        
        
        
        # lithocal = F<<pg.litho_calipers(layer1=0, layer2=1)
        # lithocal.move(lithocal.center, i.center+(140,-200))
        # lithocal = F<<pg.litho_calipers(layer1=2, layer2=1)
        # lithocal.rotate(180)
        # lithocal.move(lithocal.center, i.center+(140,-220))
        
        # lithocal = F<<pg.litho_calipers(layer1=0, layer2=1)
        # lithocal.rotate(90)
        # lithocal.move(lithocal.center, i.center+(200,-140))
        # lithocal = F<<pg.litho_calipers(layer1=2, layer2=1)
        # lithocal.rotate(-90)
        # lithocal.move(lithocal.center, i.center+(220,-140))
        
        
        line = F<<pr.route_smooth(i.ports[1], i.ports[2], width=s, layer=1)
        
        spacer = F<<pg.rectangle(size=(500,600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E<<F
    D.flatten()
    D<<E
    D.name = 'testStructureDie'
    return D


def nMemTestDie():
    D = pg.gridsweep(om.die_cell, param_y={"text1": ['NMEM ' + sub for sub in list(string.ascii_uppercase[0:7])]}, param_x={'text2':list(string.digits[1:8])}, spacing=(0,0))
    E = Device('nMem')
    loop_size = np.tile(np.linspace(0.5,3.5,7), 7)
    for i, s, k in zip(D.references, loop_size, range(0, len(loop_size))):
        F = Device('nMem_cell')
        nMem = F<<qg.memory_v4(right_extension=s)
        nMem.movex(nMem.ports[1].midpoint[0], i.ports[1].midpoint[0])
        nMem.movey(nMem.ports[3].midpoint[1], i.ports[4].midpoint[1])
        F<<pr.route_smooth(nMem.ports[1], i.ports[1], layer=1)
        F<<pr.route_smooth(nMem.ports[2], i.ports[2], layer=1)
        F<<pr.route_smooth(nMem.ports[3], i.ports[4], layer=2)
        F<<pr.route_smooth(nMem.ports[4], i.ports[3], layer=2)
        
        # if np.mod(k,2) == 0:
        bridge = F<<pg.straight((100,200),4)
        bridge.connect(bridge.ports[1], i.ports[3])
        bridge2 = F<<pg.straight((100,200),4)
        bridge2.connect(bridge2.ports[1], i.ports[4])

        spacer = F<<pg.rectangle(size=(300,600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E<<F
    # D.flatten()
    D<<E
    D.name = 'nMemTestDie'
    return D


def snspdLoopTestDie():
    D = pg.gridsweep(om.die_cell, param_y={"text1": ['SPD ' + sub for sub in list(string.ascii_uppercase[0:7])]}, param_x={'text2':list(string.digits[1:8])}, spacing=(0,0))
    E = Device('SPD')
    loop_size = np.tile(np.linspace(-2,4,7), 7)
    for i, s, k in zip(D.references, loop_size, range(0, len(loop_size))):
        F = Device('SPD_cell')
        nMem = F<<om.single_cell(loop_adjust=s)
        nMem.movex(nMem.ports[1].midpoint[0], i.ports[1].midpoint[0])
        nMem.movey(nMem.ports[3].midpoint[1], i.ports[4].midpoint[1])
        F<<pr.route_smooth(nMem.ports[1], i.ports[1], layer=1)
        F<<pr.route_smooth(nMem.ports[2], i.ports[2], layer=1)
        F<<pr.route_smooth(nMem.ports[3].rotate(180), i.ports[4], layer=2)
        F<<pr.route_smooth(nMem.ports[4].rotate(180), i.ports[3], layer=2)
        
        # if np.mod(k,2) == 0:
        bridge = F<<pg.straight((100,200),4)
        bridge.connect(bridge.ports[1], i.ports[3])
        bridge2 = F<<pg.straight((100,200),4)
        bridge2.connect(bridge2.ports[1], i.ports[4])

        spacer = F<<pg.rectangle(size=(300,600), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E<<F
    # D.flatten()
    D<<E
    D.name = 'snspdLoopTestDie'
    return D
# qp(snspdLoopTestDie())


def nTronTestDie():
    D = pg.gridsweep(om.die_cell, param_y={"text1": ['NTRON ' + sub for sub in list(string.ascii_uppercase[0:7])]}, param_x={'text2':list(string.digits[1:8])}, param_defaults = {'ports_ground':['S']},  spacing=(0,0))
    E = Device('nTron')
    choke_offset = np.tile(np.linspace(0, 3, 7), 7)
    for i, s in zip(D.references, choke_offset):
        F = Device('nTron_cell')
        nTron = F<<qg.ntron_v2(choke_offset=s)
        nTron.movex(nTron.ports[1].midpoint[0], i.ports[1].midpoint[0])
        nTron.movey(nTron.ports[3].midpoint[1], i.ports[3].midpoint[1])
        i.ports[1].width = 10
        i.ports[2].width = 10
        i.ports[3].width = 10
        i.ports[4].width = 10

        F<<pr.route_smooth(nTron.ports[1], i.ports[1], layer=1)
        F<<pr.route_smooth(nTron.ports[2], i.ports[2], layer=1)
        F<<pr.route_smooth(nTron.ports[3], i.ports[4], layer=1)
        F<<pr.route_smooth(nTron.ports[4], i.ports[3], layer=1)


        spacer = F<<pg.rectangle(size=(500,500), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E<<F
    D.flatten()
    D<<E
    D.name = 'nTronTestDie'
    return D

def nTronTestDie2():
    n=2
    N=4
    D = pg.gridsweep(om.die_cell_v3, param_y={"text1": ['NTRON ' + sub for sub in list(string.ascii_uppercase[0:N])]}, param_x={'text2':list(string.digits[1:N+1])}, param_defaults = {'size':(500*n, 500*n), 'ports':{'N':n, 'E':n, 'W':n, 'S':n}, 'ports_ground':['S']},  spacing=(0,0))
    E = Device('nTron')
    choke_offset = np.tile(np.linspace(0, 3, N), N)
    for i, s in zip(D.references, choke_offset):
        F = Device('nTron_cell')
        nTron = F<<qg.ntron_v2(choke_offset=s)
        # nTron.movex(nTron.ports[1].midpoint[0], i.ports[1].midpoint[0])
        # nTron.movey(nTron.ports[3].midpoint[1], i.ports[3].midpoint[1])
        nTron.move(nTron.center, i.center)
        # i.ports[1].width = 10
        # i.ports[2].width = 10
        # i.ports[3].width = 10
        # i.ports[4].width = 10

        F<<pr.route_smooth(nTron.ports[1], i.ports[1], layer=1, path_type='Z', length1=10, radius=15)
        F<<pr.route_smooth(nTron.ports[1], i.ports[2], layer=1, path_type='Z', length1=10, radius=15)

        F<<pr.route_smooth(nTron.ports[2], i.ports[3], layer=1, path_type='Z', length1=20, radius=15)
        
        F<<pr.route_smooth(nTron.ports[3], i.ports[5], layer=1, path_type='Z', length1=10, radius=15)
        F<<pr.route_smooth(nTron.ports[3], i.ports[6], layer=1, path_type='Z', length1=10, radius=15)

        F<<pr.route_smooth(nTron.ports[4], i.ports[7], layer=1, path_type='Z', length1=60, radius=15)
        F<<pr.route_smooth(nTron.ports[4], i.ports[8], layer=1, path_type='Z', length1=60, radius=15)

        # F<<pr.route_smooth(nTron.ports[2], i.ports[2], layer=1)
        # F<<pr.route_smooth(nTron.ports[3], i.ports[4], layer=1)
        # F<<pr.route_smooth(nTron.ports[4], i.ports[3], layer=1)


        spacer = F<<pg.rectangle(size=(550,550), layer=3)
        spacer.move(spacer.center, i.center)
        F = pg.union(F, by_layer=True)
        # F.flatten()
        E<<F
    # D.flatten()
    D<<E
    D.name = 'nTronTestDie'
    return D
# D=nTronTestDie2()
# qp(nTronTestDie2())

def snspdTestDie():
    D = pg.gridsweep(om.die_cell, param_y={"text1": ['SPD ' + sub for sub in list(string.ascii_uppercase[0:7])]}, param_x={'text2':list(string.digits[1:8])}, param_defaults = {'ports_ground':['S']},  spacing=(0,0))
    E = Device('SNSPD')
    snspd_area = np.tile(np.linspace(10, 250, 7), 7)
    for i, s in zip(D.references, snspd_area):
        F = Device('snspd_cell')
        spd = F<<pg.snspd_expanded(wire_width=0.1, wire_pitch=0.4, size=(s,s), layer=1)
        spd.rotate(-90)
        spd.move(spd.center, i.center)

        F<<pr.route_smooth(spd.ports[1], i.ports[1], layer=1)
        F<<pr.route_smooth(spd.ports[2], i.ports[2], layer=1)

        spacer = F<<pg.rectangle(size=(500,500), layer=3)
        spacer.move(spacer.center, i.center)
        F.flatten()
        E<<F
    D.flatten()
    D<<E
    D.name = 'snspdTestDie'
    return D







def nMemArray(n=8, text1='A', text2='B'):
    D = om.die_cell_v3(ports = {'N':n, 'E':n, 'W':n, 'S':n},ports_ground=['E','S'], size=(300*n,300*n), size2=(600,600), text1=text1, text2=text2)
    F = Device('nmem_array_cell')
    nmem = F<<om.memory_array(n,n)
    nmem.move(nmem.center, D.center)

    layers = np.append(np.tile(1, 2*n), np.tile(2, 2*n))

    for i in range(1,2*n+1):
        F<<pr.route_smooth(nmem.ports[i], D.ports[i], path_type='Z', length1=50, length2=20, layer=1, radius=1)
        
    heaterIntsizeX = abs(D.ports[1].midpoint[0]-D.ports[2*n].midpoint[0])-80
    heaterIntsizeY = abs(D.ports[2*n].midpoint[1])+100
    heaterIntsize = np.array([heaterIntsizeX, heaterIntsizeY])
    heaterInt = F<<pg.compass_multi(heaterIntsize, ports={"W":n, 'E':n})
    heaterInt_ports = list(heaterInt.ports.values())
    heaterInt_ports.sort(key=port_norm, reverse=True)
    
    nmem_portlist = list(nmem.ports.values())
    nmem_portlist2 = nmem_portlist[2*n:4*n]
    nmem_portlist2 = port_nearest(heaterInt_ports, nmem_portlist2)

    for i in range(0,2*n):
        F<<pr.route_sharp(nmem_portlist2[i], heaterInt_ports[i].rotate(180), length1=10, length2=30, width=(nmem_portlist2[i].width, 10), path_type='Z', layer=2)
        extension = F<<pg.straight(size=(10,20), layer=2)
        extension.connect(extension.ports[1], heaterInt_ports[i].rotate(180))
    padPorts = list(D.ports.values())
    padPorts = padPorts[2*n:4*n]
    padPorts.sort(key=port_norm, reverse=True)
    heaterInt_ports = port_nearest(padPorts, heaterInt_ports)
    for i in range(0,2*n):
        F<<pr.route_sharp(padPorts[i], heaterInt_ports[i], length1=30, length2=30, width=padPorts[i].width, path_type='Z', layer=4)
            
    F.remove(heaterInt)
    spacerLoc1 = D.ports[1].midpoint+np.array([-20,10])
    spacerLoc2 = D.ports[2*n].midpoint+np.array([20,-10])
    spacer = F<<pg.rectangle(size=spacerLoc1-spacerLoc2, layer=3)
    spacer.move(spacer.center, D.center)
    F<<D
    return F

def snspdArray(n=8, text1='A', text2='B'):
    D = om.die_cell_v3(ports={'N':n, 'E':n, 'W':n, 'S':n},ports_ground=['E','S'], size=(300*n, 300*n), size2=(650,650), ground_width=300, text1=text1, text2=text2)

    F = Device('snspd_array')
    spd = F<<om.single_bit_array(n,n)
    spd.move(spd.center, D.center)

    for i in range(1,2*n+1):
        F<<pr.route_smooth(spd.ports[i], D.ports[i], path_type='Z', length1=50, length2=20, layer=1, radius=1)


    heaterIntsizeX = abs(D.ports[1].midpoint[0]-D.ports[2*n].midpoint[0])-80
    heaterIntsizeY = abs(D.ports[2*n].midpoint[1])+100
    heaterIntsize = np.array([heaterIntsizeX, heaterIntsizeY])
    heaterInt = F<<pg.compass_multi(heaterIntsize, ports={"W":n, 'E':n})
    heaterInt_ports = list(heaterInt.ports.values())
    heaterInt_ports.sort(key=port_norm, reverse=True)
    
    spd_portlist = list(spd.ports.values())
    spd_portlist2 = spd_portlist[2*n:4*n]
    spd_portlist2 = port_nearest(heaterInt_ports, spd_portlist2)

    for i in range(0,2*n):
        F<<pr.route_sharp(spd_portlist2[i].rotate(180), heaterInt_ports[i].rotate(180), length1=10, length2=30, width=(spd_portlist2[i].width, 10), path_type='Z', layer=2)
        extension = F<<pg.straight(size=(10,20), layer=2)
        extension.connect(extension.ports[1], heaterInt_ports[i].rotate(180))
    padPorts = list(D.ports.values())
    padPorts = padPorts[2*n:4*n]
    padPorts.sort(key=port_norm, reverse=True)
    heaterInt_ports = port_nearest(padPorts, heaterInt_ports)
    for i in range(0,2*n):
        F<<pr.route_sharp(padPorts[i], heaterInt_ports[i], length1=30, length2=30, width=padPorts[i].width, path_type='Z', layer=4)
        
    F.remove(heaterInt)
   
    spacerLoc1 = D.ports[1].midpoint+np.array([-20,10])
    spacerLoc2 = D.ports[2*n].midpoint+np.array([20,-10])
    spacer = F<<pg.rectangle(size=spacerLoc1-spacerLoc2, layer=3)
    spacer.move(spacer.center, D.center)
    
    # F.flatten()
    # E<<F
    # D.flatten()
    D<<F
    D.name = 'snspdArrayDie'
    return D


def htron4p(channel_width=0.1, heater_width=0.1, text1='A', text2='B'):
    n=4
    D = pg.gridsweep(om.die_cell_v3, param_y={"text1": list(string.ascii_uppercase[0:3])}, param_x={'text2':list(string.digits[1:4])}, param_defaults = {'ports':{'N':n, 'E':n, 'W':n, 'S':n}, 'ports_ground':['E','S'], 'ground_width':300, 'size':(400*n, 400*n), 'size2':(650,650) },  spacing=(0,0))
    # E = Device('SNSPD')
    E = Device()
    channel_widths = np.tile([0.1, 0.5, 1], 3)
    heater_widths = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1, 1, 1]
    for i, s, r in zip(D.references, channel_widths, heater_widths):
        F = Device()
        wire = F<<pg.straight(size=(s,10), layer=1)
        wire.rotate(90)
        wire.move(wire.center, i.center)
        # wire.movey(100)
        taper = pg.optimal_step(s, 4, symmetric=True, layer=1, num_pts=200)
        ta1 = F<<taper
        ta2 = F<<taper
        
        ta1.connect(ta1.ports[1], wire.ports[1])
        ta2.connect(ta2.ports[1], wire.ports[2])
        
        F<<pr.route_smooth(ta1.ports[2], i.ports[1], path_type='Z', length1=1, radius=1.1, layer=1)
        F<<pr.route_smooth(ta1.ports[2], i.ports[2], path_type='Z', length1=1, radius=1.1, layer=1)
        
        F<<pr.route_smooth(ta2.ports[2], i.ports[3], path_type='Z', length1=1, radius=1.1, layer=1)
        F<<pr.route_smooth(ta2.ports[2], i.ports[4], path_type='Z', length1=1, radius=1.1, layer=1)
        
        
        heater = F<<pg.straight(size=(r, 5), layer=2)
        heater.move(heater.center, wire.center)
        taper = pg.optimal_step(r, 4, symmetric=True, layer=2, num_pts=200)
        
        ta3 = F<<taper
        ta4 = F<<taper
        
        ta3.connect(ta3.ports[1], heater.ports[1])
        ta4.connect(ta4.ports[1], heater.ports[2])
        
        tee = pg.tee(size=(8,4), stub_size=(4,1), taper_type = 'fillet', layer=2)
        t1 = F<<tee
        t1.connect(t1.ports[3], ta3.ports[2])
        t2 = F<<tee
        t2.connect(t2.ports[3], ta4.ports[2])
        
        
        F<<pr.route_smooth(t1.ports[1], i.ports[12], path_type='U', length1=2,  radius=.5, layer=2)
        F<<pr.route_smooth(t1.ports[2], i.ports[11], path_type='U', length1=2,  radius=.5, layer=2)
        
        F<<pr.route_smooth(t2.ports[1], i.ports[10], path_type='U', length1=2, radius=.5, layer=2)
        F<<pr.route_smooth(t2.ports[2], i.ports[9], path_type='U', length1=2, radius=.5, layer=2)
        
        spacerLoc1 = i.ports[1].midpoint+np.array([-20,10])
        spacerLoc2 = i.ports[2*n].midpoint+np.array([20,-10])
        spacer = F<<pg.rectangle(size=spacerLoc1-spacerLoc2, layer=3)
        spacer.move(spacer.center, i.center)
        
        for k in range(9,13):
                bridge = F<<pg.straight((50,200),4)
                bridge.connect(bridge.ports[1], i.ports[k])
                bridge.movex(-30)
                
                
        F = pg.union(F, by_layer=True)
        E<<F
        
        # t1 = D<<tee
    E<<D
    return E

def nTronTestImag():
    D = Device()
    D<<pg.gridsweep(qg.ntron, spacing=(10,10), param_x={'choke_w': np.linspace(0.01, 0.1, 10)}, param_y={'channel_w': np.tile(0.2, 20)})
    
    return D
    
# qp(nTronTestImag())

#%% 

D = Device()
n=2
text1='a'
text2='1'
pads = D<<om.die_cell_v3(ports={'N':n, 'E':n, 'W':n, 'S':n},ports_ground=['E','S'], size=(250*2*n, 250*2*n), size2=(650,650), ground_width=300, text1=text1, text2=text2)







#%%
def nMemArray4():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    N=4
    dut = D<<pg.gridsweep(nMemArray, spacing=(0,0), param_x={'text1': list(string.ascii_uppercase[0:N])}, param_y={'text2': list(string.digits[1:N+1])}, param_defaults={'n':4})
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D

def nMemArray8():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    N=2
    dut = D<<pg.gridsweep(nMemArray, spacing=(0,0), param_x={'text1': list(string.ascii_uppercase[0:N])}, param_y={'text2': list(string.digits[1:N+1])}, param_defaults={'n':8})
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D

def snspdArray4():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    N=4
    dut = D<<pg.gridsweep(snspdArray, spacing=(0,0), param_x={'text1': list(string.ascii_uppercase[0:N])}, param_y={'text2': list(string.digits[1:N+1])}, param_defaults={'n':4})
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D

def snspdArray8():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    N=2
    dut = D<<pg.gridsweep(snspdArray, spacing=(0,0), param_x={'text1': list(string.ascii_uppercase[0:N])}, param_y={'text2': list(string.digits[1:N+1])}, param_defaults={'n':8})
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D

def nMemLoop():
     D = Device()
     rec = D<<pg.rectangle((10000,10000), layer=99)
     marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
     marks.move(marks.center, rec.center)
     dut = D<<nMemTestDie()
     dut.move(dut.center, rec.center)
     D.remove(rec)
     return D
    

def snspdLoop():
     D = Device()
     rec = D<<pg.rectangle((10000,10000), layer=99)
     marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
     marks.move(marks.center, rec.center)
     dut = D<<snspdLoopTestDie()
     dut.move(dut.center, rec.center)
     D.remove(rec)
     return D

def htron4pGrid():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    dut = D<<htron4p()
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D
 
def ntron4pGrid():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    dut = D<<nTronTestDie2()
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D  
    
def snspdTest():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    dut = D<<snspdTestDie()
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D  

def testStructures():
    D = Device()
    rec = D<<pg.rectangle((10000,10000), layer=99)
    marks = D<<qg.alignment_marks(locations=((-4500, -4500), (-4500, 4500), (4500, -4500), (4500, 4500)), layer=0)
    marks.move(marks.center, rec.center)
    dut = D<<testStructureDie()
    dut.move(dut.center, rec.center)
    D.remove(rec)
    return D  








#%% 
A = Device('wafer')

A<<pg.ring(radius=100e3/2, layer=99)

Ndie = 45
 # param_x={'text2':list(string.digits[1:8])}, param_defaults = {'ports_ground':['S']},  spacing=(0,0))
# die = pg.basic_die(size=(10e3, 10e3), die_name=)
# die_list = np.tile(die, 36)

die_array = pg.gridsweep(om.basic_die, param_x={'text1': list(string.ascii_uppercase[0:7])}, param_y={'text2': list(string.digits[1:8])}, param_defaults={'size': (10e3, 10e3), 'text_size': 400, 'text_location':'S'}, spacing=(0,0))


die_array.move(die_array.center, (0,0))

A<<die_array


tstart = time.time()

for i in [10, 38]:
    a = A<<nMemArray4()
    a.move(a.center, die_array.references[i].center)
    
for i in [17, 31]:
    a = A<<nMemArray8()
    a.move(a.center, die_array.references[i].center)

for i in [22, 26]:
    a = A<<snspdArray4()
    a.move(a.center, die_array.references[i].center)

for i in [23, 25]:
    a = A<<snspdArray8()
    a.move(a.center, die_array.references[i].center)
    
for i in [16, 32]:
    a = A<<nMemLoop()
    a.move(a.center, die_array.references[i].center)
    
for i in [18, 30]:
    a = A<<snspdLoop()
    a.move(a.center, die_array.references[i].center)

for i in [15, 19, 24, 29, 33]:
    a = A<<htron4pGrid()
    a.move(a.center, die_array.references[i].center)
    
tend = time.time()
print(tend-tstart)
qp(A)



#%% 

B = Device('wafer')

B<<pg.ring(radius=100e3/2, layer=99)

Ndie = 45
 # param_x={'text2':list(string.digits[1:8])}, param_defaults = {'ports_ground':['S']},  spacing=(0,0))
# die = pg.basic_die(size=(10e3, 10e3), die_name=)
# die_list = np.tile(die, 36)

die_array = pg.gridsweep(om.basic_die, param_x={'text1': list(string.ascii_uppercase[0:7])}, param_y={'text2': list(string.digits[1:8])}, param_defaults={'size': (10e3, 10e3), 'text_size': 400, 'text_location':'S'}, spacing=(0,0))


die_array.move(die_array.center, (0,0))

B<<die_array


    
for i in [16, 18, 30, 32]:
    a = B<<ntron4pGrid()
    a.move(a.center, die_array.references[i].center)
    
for i in [17, 23, 25, 31]:
    a = B<<snspdTest()
    a.move(a.center, die_array.references[i].center)

for i in [10, 22, 24, 26, 38]:
    a = B<<testStructures()
    a.move(a.center, die_array.references[i].center)

qp(B)
#%% 

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



