"""
Uncertainty Optimization for ONERA M6 full turbulence
Author : X.J.
"""

# ======================================================================
#         Import modules
# ======================================================================
# rst Imports (beg)
import os
import numpy as np
import argparse
import ast
import pygpc
import time
from mpi4py import MPI
from baseclasses import AeroProblem
from adflow import ADFLOW
from pygeo import *
from pyoptsparse import Optimization, OPT
# from pywarp import *
from multipoint import *
from collections import OrderedDict
# from pycalc import AeroTransi
# from pyUncertainty import UncertaintyOptimization
from idwarp import USMesh
from pyspline import *
from pyspline import Curve
from Uncertaintypackage.pyUncertainty import UncertaintyOptimization
from Uncertaintypackage.Geodis import Geo_stochastic
import pprint
import copy
# rst Imports (end)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--gridFile", type=str, default="rae2822.cgns")
parser.add_argument("--opt", type=str, default="SNOPT", choices=["SLSQP", "SNOPT"])
parser.add_argument("--optOptions", type=ast.literal_eval, default={}, help="additional optimizer options to be added")
args = parser.parse_args()

# ======================================================================
#         Create multipoint communication object
# ======================================================================
# rst multipoint (beg)

MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
gcomm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

# gcomm = MPI.COMM_WORLD
# print('#1 gcomm.rank:',gcomm.rank)
npUncert = 1
npAero = gcomm.size - npUncert


# Creat aero/transition comms
comm, flags = createGroups([npUncert, npAero], comm=gcomm)
aeroID = 1
UncertID = 0

print('comm', comm.rank, flags)
# print('#2 gcomm.rank:',gcomm.rank)
# exit()
if not os.path.exists(args.output):
    if gcomm.rank == 0:
        os.mkdir(args.output)
# rst multipoint (end) 
# ======================================================================
#         Specify parameters for caculation
# ======================================================================
# rst parameters (beg)
# cL constraint
mycl = 0.25929
mycm = 0.18257
# angle of attack
alphamean = 3.06
# mach number
# machmean = 0.19
machmean = 0.8395
# reynold number
# reynolds = 5.6e6
reynolds = 11.72e6
reynoldsLength = 0.64607
# reference area
areaRef = 0.772893541
# reference chord
chordRef = 0.64607
# temperature K
# T = 286.72
T = 255.56
# Turbulence intensity
Tu = 0.07 # 0.07% 
spanDirection = 'z'
# rst parameters (end)
grid_file ='../'+args.gridFile

# # ======================================================================
# #         ADflow Set-up
# # ======================================================================
# rst adflow (beg)
aeroOptions = {
    # I/O Parameters
    'gridFile':grid_file,
    'outputDirectory':args.output,
    'monitorvariables':['resrho','cl','cpu','cd','cmz','resturb','cdp','cdv'],
    'volumevariables':['resrho','Intermittency', 'cp', 'mach', 'temp', 'rhoe'],
    'surfacevariables':['cp','vx', 'vy','vz', 'mach','cfx', 'mach', 'rho', 'p', 'temp', 'cf', 'yplus','blank'],
    'writeTecplotSurfaceSolution':True,
    'writeVolumeSolution':False,
    'writeSurfaceSolution': False,
	# 'solRestart' : True,
	#'restartFile' : './fc_000_vol.cgns',

    # Physics Parameters
    'equationType':'RANS',

    # Solver Parameters
    'CFL':1.5,
    'CFLCoarse':1.25,
    'MGCycle':'SG',
    
    'rkreset': True,
    'nrkreset' : 100,

    # ANK Solver Parameters
    'useANKSolver':True,
    'nsubiterturb':15,
    'anksecondordswitchtol':1e-6, # increased for 30 deg
    # 'ankcflfactor': 4.0,
    # 'ankcoupledswitchtol':1e-5,
    # ANK yayun's setting
	'useANKSolver':True,
    # 'ankuseturbdadi':False,
    # 'ankturbkspdebug':True,

	'ankstepfactor' : 0.5,

	#'ankcoupledswitchtol' : 1e-6,
	'ankmaxiter' : 60,

    # NK Solver Parameters
    'useNKSolver':True,
    'nkswitchtol':1e-9,
    'nkadpc':True,
    # 'nkasmoverlap': 3, # for highly parallel
    'nkinnerpreconits': 2,
    'nkjacobianlag': 3,
    'nkouterpreconits': 3,
    'nkpcilufill': 2,
    'nksubspacesize': 100,
    # ANK Solver Parameters
    'useANKSolver':True,
    'anklinearsolvetol':0.1, 
    'ankswitchtol':1e-2,
    'anksubspacesize':200,
    'nsubiterturb':40,
    'smoother':'Runge-Kutta',
    'resaveraging':'alternate',
    'smoothparameter':1.5,  

    # Termination Criteria
    'L2Convergence':1e-8,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':30000,
    'useblockettes':False,
    
    #Turb stuff
    # 'useqcr':True, # go 2 NASA tmr for more info (closer to exp??)
    # Following 3 give defalt SA-noft2 (fully turb sim, in lit)
    # 'eddyvisinfratio':.210438,
    'useft2SA' : False,
    # 'turbulenceproduction' : 'vorticity',
    

    # using chenyifu's setting
    'turbResScale': 10000.0,
    'useNKSolver': True,
    'MGCycle': 'sg',
    'NKASMOverlap': 4,
    'NKInnerPreconIts': 3,
    'NKJacobianLag': 5,
    'NKOuterPreconIts': 3,
    'NKPCILUFill': 3,
    'NKSubspaceSize': 300,
    'NKSwitchTol': 1e-06,
    'adjointMaxIter': 5000,
    'adjointSubspaceSize': 400,
    'ADPC': True,
    'ASMOverlap': 3,
    'ILUFill': 3,
    'nSubiterTurb': 10,
    # if use transition
    # 'ntransition':False, 
    # 'transi2dim':True,
    # 'useintermittency':True,
	# 'BoundaryLayerThickness':0.06,
    # 'usexyzlstate':True,
    # 'rkreset':True,
    # 'nrkreset':100,

    # # Adjoint Parameters
    'setMonitor':False,
    # 'applyadjointpcsubspacesize':15,
    # 'adjointL2Convergence':1e-8,
    # 'ADPC':True,
    # 'adjointMaxIter': 6000,
    # 'adjointSubspaceSize':150, 
    # 'ILUFill':2,
    # 'ASMOverlap':1,
    # 'outerPreconIts':3,
    # "NKSubSpaceSize": 400,
    "adjointSolver": "GMRES",
    "adjointL2Convergence": 1e-7,
    "ADPC": True,
    "adjointMaxIter": 5000,
    "adjointSubspaceSize": 400,
    "ILUFill": 3,
    "ASMOverlap": 3,
    "outerPreconIts": 3,
    "NKSubSpaceSize": 400,
    "frozenTurbulence": False,
    "restartADjoint": False,
}


# Create solver
# if flags[aeroID]:
if 1==1:
    print("Begin aero setting, gcomm.rank=",gcomm.rank)
    CFDSolver = ADFLOW(options=aeroOptions, comm=comm)
    # print('gcomm.rank:',gcomm.rank)
    print('comm.rank:',comm.rank)
    print('*****')
    print('CFDSolver.comm.rank', CFDSolver.comm.rank)
    CFDSolver.addLiftDistribution(200, "z")
# if flags[UncertID]:
#     CFDSolver = None


# ======================================================================
#         Uncertainty Optimization set-up
# ======================================================================
name = 'ONERAm6'
# define parameters
dim = 11
Geo_NumMode = 9
parameters = OrderedDict()
parameters["mach"] = pygpc.Norm(pdf_shape=[machmean, 0.015])
parameters["alpha"]  = pygpc.Norm(pdf_shape=[alphamean, 0.15])
# parameters["nCritTS"] = pygpc.Beta(pdf_shape=[4.3,3], pdf_limits=[4, 13])
for i in range(Geo_NumMode):
    parameters["geo_"+str(i)] = pygpc.Norm(pdf_shape=[0.0, 1.0])
###############
# gPC options #
###############
options = dict()
save_session_format = '.pkl'
fn_results = 'RAE2822'

# setup grid
options["grid"] = pygpc.Random
options["grid_options"] = None

# setup order information
p = 2
options["order"] = [p]*dim
options["order_max"] = p
options["order_max_norm"] = 1.0
options["interaction_order"] = 10

# setup solver
# options["solver"] = "LarsLasso"
# options["solver"] = "NumInt"
options["solver"] = "Moore-Penrose"
# setup method (only for static algorithm)
# options["method"] = "quad"
options["method"] = "reg"

# setup cpu parallels
options["n_cpu"] = 0

# save format
options["save_session_format"] = save_session_format
options["fn_results"] = fn_results

# specific options
# no grad enh
# options["gradient_enhanced"] = False
# grad enh
options["gradient_enhanced"] = True
options["gradient_calculation"] = "External"
options["exGrad_options"] = "TestFunc"
options["hessianmethod"] = "SR1" 
options["hessianswitchtol"] = 0.1
options["hessiannstart"] = 30

options['asymptotic'] = 'Hermite'

# options["error_type"] = "nrmsd"
options["error_type"] = "loocv"
options["eps"] = 0.001
options["matrix_ratio"] = 2.0
options["backend"] = "omp"        # we now use all CPU cores to calculate the gPC matrices (makes things considerably faster)
samplemethod = 'qr'

# ======================================================================
#         Geometric Design Variable Set-up
# ======================================================================

# rst dvgeo (beg)
# Create DVGeometry object
FFD_file = '../m6_ffd.x'

# # if flags[aeroID]:
# exec(open('./common_files/setup_geometry.py').read())
# ======================================================================
#         Geometry set-up
# ======================================================================

wing_vol=[0]
ind = []
DVGeo = DVGeometry(FFD_file)
for ivol in wing_vol:
    for i in range(1,DVGeo.FFD.vols[ivol].nCtlu-1):
        for j in range(DVGeo.FFD.vols[ivol].nCtlv):
            for k in range(DVGeo.FFD.vols[ivol].nCtlw):
                ind.append(DVGeo.FFD.topo.lIndex[ivol][i,j,k])
ind = geo_utils.unique(ind)
wing_PS = geo_utils.PointSelect('list',ind)
nval = DVGeo.addLocalDV(dvName='wingshapevars',lower = -2, upper = 2,scale = 100.0,axis ='y',pointSelect=wing_PS)

x_le_wing_twist = [
    [0.001,0.000,0.0001],
    [0.236,0,   0.4083],
    [0.474,0,0.8152],
    [0.69,0,1.194]
]
nTwist = 4
nTwistvalue=nTwist-1
c_wing = Curve(X=x_le_wing_twist,k=2)
DVGeo.addRefAxis('wing_axis',c_wing,volumes=wing_vol[0])
def twistWingfunc(val,geo):
    for i in range(nTwistvalue):
        geo.rot_z['wing_axis'].coef[i+1]=val[i]
#because we do not need rotation of the slice z=0, for the aoa can change
DVGeo.addGlobalDV(dvName="twist_wing",value=np.zeros(nTwistvalue),func=twistWingfunc,lower=np.array([-1.,-1.,-1.]),upper=np.array([1.,1.,1.]),scale=1.0)

# ind2 = []
# for ivol in wing_vol:
#     for i in [0,DVGeo.FFD.vols[ivol].nCtlu-1]:
#         for j in range(DVGeo.FFD.vols[ivol].nCtlv):
#             for k in range(DVGeo.FFD.vols[ivol].nCtlw):
#                 ind2.append(DVGeo.FFD.topo.lIndex[ivol][i,j,k])
# ind2 = geo_utils.unique(ind2)
# wing_PS2 = geo_utils.PointSelect('list',ind2)
# nval2 = DVGeo.addLocalDV(dvName='wingshapevars2',lower = -1e-6, upper = 1e-6,scale = 100.0,axis ='y',pointSelect=wing_PS2)
if gcomm.rank == 0:
    print("DVGeo setting done")
# if flags[aeroID]:
#     CFDSolver.setDVGeo(DVGeo)
#Geo_NumMode = 6
Geo_stochastic1 = Geo_stochastic(DVGeo = DVGeo,
                                NumMode = Geo_NumMode,
                                parageobegin = 2,
                                dim_select = 3,
                                spandirection = 'z',
                                MeanValue = 0.0,
                                StdValue = 1.0,
                                ref_dis = 0.64607,
                                select_Vol_list=[0],
                                select_LocalDV_list=['wingshapevars'],
                                disturb_scale = 1.248033e-3,
                                )

print("Done Geosto setting, gcomm.rank=",gcomm.rank)
if flags[aeroID]:
    print("Begin CFDSolver geometry setting, comm=",comm.rank)
    span = 1.1963
    pos  = np.array([0.01,0.2,0.44,0.65,0.8,0.9,0.95,0.99])*span
    CFDSolver.addSlices(spanDirection,pos,sliceType='absolute')


    # Add DVGeo object to CFD solver
    CFDSolver.setDVGeo(DVGeo)
    if comm.rank == 0:
        print("Done CFDSolver geometry setting")
# rst dvgeo (end)
# ======================================================================
#         Uncertainty Optimization set-up
# ======================================================================

if flags[UncertID]:
    UncertOpt = UncertaintyOptimization(name=name,
                                    AeroDVkey = ['alpha'],
                                    GeoDVname=['wingshapevars','twist'],
                                    outputdir=args.output,
                                    parameters=parameters,
                                    samplefrozen=True,
                                    samplemethod=samplemethod,
                                    Geo_sto_list=[Geo_stochastic1],
                                    gcomm=comm,
                                    # CFDSolver=None,
                                    verbose=False)
    UncertOpt.n_out = 3
    UncertOpt.dim = dim
    UncertOpt.evalFuncs = ['cl','cd','cmz']
    if comm.rank == 0:
        print("Done UncertOpt setting")
    # print('#1 UncertOpt.gcomm.rank ', UncertOpt.gcomm.rank)

# rst adflow (end)
# Determine aroot and uroot
data = -1
data = gcomm.allgather(data)
print('data',data)

data = -1
# exit()
if flags[aeroID]:
    # print('CFDSolver.comm.rank ', CFDSolver.comm.rank)
    data = comm.rank
    print(data)
data =gcomm.allgather(data)
print('#1',data)
# print('#2',data)

aroot = data.index(0)
print('aroot',aroot)
# exit()
data = -1

if flags[UncertID]:
    print('#2 UncertOpt.gcomm.rank ', comm.rank)
    data = comm.rank
print('#3 ',data)
data = gcomm.allgather(data)
print('#4 ',data)
uroot = data.index(0)
print('uroot',uroot)


# ======================================================================
#         Sampling and AeroProblems Set-up
# ======================================================================


benchmarkProblem = AeroProblem(name=name+'_benchmark',
                            mach = machmean,
                            alpha = alphamean,
                            reynolds=reynolds,
                            reynoldsLength = reynoldsLength,
                            T = T ,
                            areaRef=areaRef,
                            chordRef=chordRef,
                            xRef=0.0,yRef=0.0,zRef=0.0,
                            evalFuncs=['cl','cd','cmz'])
benchmarkProblem.addDV('alpha',value=alphamean,lower=1.0,upper=4.0,scale=0.3)
n_grid = None
samples = None
if flags[UncertID]:
    # sampling
    if samplemethod == 'qr':
        UncertOpt.n_basis, UncertOpt.n_grid, UncertOpt.samples, UncertOpt.grid = UncertOpt.GenerateSamples(options=options,n_grid =int( 400000))
    else:
        UncertOpt.n_basis, UncertOpt.n_grid, UncertOpt.samples, UncertOpt.grid = UncertOpt.GenerateSamples(options=options)
    n_grid = UncertOpt.n_grid
    samples = UncertOpt.samples

# broadcast variable
n_grid = gcomm.bcast(n_grid,root=uroot)
samples = gcomm.bcast(samples,root=uroot)
if gcomm.rank == 0:
    pprint.pprint(vars(UncertOpt))
# set uncertainty problems
aeroProblems = []
print('n_grid:',n_grid)
print('samples:',samples.shape)
for i in range(n_grid):
    ap = AeroProblem(name=name+'_'+samplemethod+str(i),
                    mach = samples[i,0],
                    alpha = samples[i,1],
                    reynolds=reynolds,
                    reynoldsLength = reynoldsLength,
                    T = T ,
                    areaRef=areaRef,
                    chordRef=chordRef,
                    xRef=0.0,yRef=0.0,zRef=0.0,
                    evalFuncs=['cl','cd','cmz'])

    for key in parameters:
        key = str(key).lower()
        # if key == 'alpha' or key == 'mach':
        if key == 'mach':
            ap.addDV(key, value=parameters[key].mean, lower=parameters[key].mean, upper=parameters[key].mean, scale=1.0)
        if key == 'alpha':
            ap.addDV(key, value=samples[i,1], lower=samples[i,1]-5.0, upper=samples[i,1]+5.0, scale=1.0)
    
    aeroProblems.append(ap)



# ======================================================================
#         DVConstraint Setup
# ======================================================================
if flags[aeroID]:
    exec(open('./common_files/setup_constraints.py').read())
if gcomm.rank == aroot:
    fileName = os.path.join(args.output, "constraints.dat")
    DVCon.writeTecplot(fileName)
# rst dvcon (end)

# ======================================================================
#         Mesh Warping Set-up
# ======================================================================
# rst warp (beg)
meshOptions = {
    "gridFile": grid_file,
    # 'warpType':'algebraic'
    'filetype':'CGNS'
    }

if flags[aeroID]:
    mesh = USMesh(options=meshOptions, comm=comm)
    # mesh = USMesh(options=meshOptions, comm=comm)
    CFDSolver.setMesh(mesh)
# rst warp (end)

# ======================================================================
#         Functions:
# ======================================================================
def updateSampleVars(dvDict,AeroProblem,DVGeometry=None,**kwargs):

    dvDict_disturbedi = OrderedDict()
    dvDict_disturbedi.update(AeroProblem.getDesignVars())
    for dvName_ in dvDict:

        if dvName_ in AeroProblem.DVs:

            if (dvDict[dvName_]!= AeroProblem.DVs[dvName_].value):
                print("Warning: Aero variable {} in dvDict is inconsistent with the one in AeroProblem. Using the variable in AeroProblem instead.".format(dvName_))


        if DVGeometry is not None:
            # if dvName_ in DVGeometry.DV_listLocal:

            #         # if (dvDict[dvName_]!= DVGeometry.DV_listLocal[dvName_].value):
            #         #     print("warnning: Geo variable {} in dvDict are inconsistent with ones in Geo_sto_list. Use variables in Geo_sto_list instead.".format(dvName_))
            #     vals_to_update={}
            #     vals_to_update[dvName_] = DVGeometry.DV_listLocal[dvName_].value
            #     dvDict_disturbedi.update(vals_to_update)
            dvDict_disturbedi.update(DVGeometry.getValues())

        for key_, dvobject in kwargs.items(): 
            if dvName_ in dvobject:
                vals_to_update={}
                vals_to_update[dvName_] = dvobject[dvName_].value
                dvDict_disturbedi.update(vals_to_update)

    return dvDict_disturbedi
# rst funcs (beg)

def cruiseFuncs(x):

    if flags[UncertID]:
        UncertOpt.iOpt += 1
        print('+----------------+')
        print('|  iOpt = %-7i|'%(UncertOpt.iOpt))
        print('+----------------+')
        print('Design Variables:')
        print(x)
        UncertOpt.Optworkdir = os.path.join (UncertOpt.outputdir,'iOpt' + str(UncertOpt.iOpt))

    if flags[UncertID] and not os.path.exists(UncertOpt.Optworkdir):
        os.mkdir(UncertOpt.Optworkdir)

    # Set geometry design vars
    DVGeo.setDesignVars(x)
    Geo_stochastic1.update_tran_matrix(x)
    # init 
    funcs = {}
    benchmarktmp_funcs = {}
    sampletmp_funcs = {}
    benchmarktmp_Sens = {}
    sampletmp_Sens = {}

    if flags[UncertID]:
        UncertOpt.benchmark_funcs = None
        UncertOpt.benchmark_Sens = None
        UncertOpt.funcs = None
        UncertOpt.funcsSens = None

    # cal
    if flags[aeroID]:
        benchmarkProblem.setDesignVars(x)
        CFDSolver(benchmarkProblem)
        CFDSolver.evalFunctions(benchmarkProblem, benchmarktmp_funcs)
        CFDSolver.evalFunctionsSens(benchmarkProblem,benchmarktmp_Sens)
        dvDict_disturbed=[]
        for i in range(n_grid):
            x["alpha_"+aeroProblems[i].name] = x["alpha_"+benchmarkProblem.name] + samples[i,1] - alphamean
            aeroProblems[i].setDesignVars(x)
            Geo_stochastic1.setGeodisturb(activated_sample=samples,ti = i,parageobegin=2)
            
            dvDict_disturbed.append(updateSampleVars(dvDict=x,AeroProblem=aeroProblems[i],DVGeometry=DVGeo))
            CFDSolver(aeroProblems[i])
            CFDSolver.evalFunctions(aeroProblems[i], sampletmp_funcs)
            CFDSolver.evalFunctionsSens(aeroProblems[i],sampletmp_Sens)
            Geo_stochastic1.restoreGeodisturb()
    # send data from aero group processor to uncertainty group processor
    if gcomm.rank == aroot:
        gcomm.send(dvDict_disturbed, dest=uroot,tag=11)
        gcomm.send(benchmarktmp_funcs, dest=uroot,tag=11)
        gcomm.send(benchmarktmp_Sens, dest=uroot,tag=11)
        gcomm.send(sampletmp_funcs, dest=uroot,tag=11)
        gcomm.send(sampletmp_Sens, dest=uroot,tag=11)
    elif gcomm.rank == uroot:
        dvDict_disturbed = gcomm.recv(source=aroot, tag=11)
        benchmarktmp_funcs = gcomm.recv(source=aroot, tag=11)
        benchmarktmp_Sens = gcomm.recv(source=aroot, tag=11)
        sampletmp_funcs = gcomm.recv(source=aroot, tag=11)
        sampletmp_Sens = gcomm.recv(source=aroot, tag=11)
    if flags[UncertID] :
        UncertOpt.dvDict_disturbed = dvDict_disturbed
        UncertOpt.benchmark_funcs = benchmarktmp_funcs
        UncertOpt.benchmark_Sens = benchmarktmp_Sens
        UncertOpt.funcs = sampletmp_funcs
        UncertOpt.funcsSens = sampletmp_Sens

        print('funcs in benchmark:')
        print(UncertOpt.benchmark_funcs)
        print('funcsSens in benchmark:')
        print(UncertOpt.benchmark_Sens)
        print('funcs for PCE:')
        print(UncertOpt.funcs)
        print('funcsSens for PCE:')
        print(UncertOpt.funcsSens)

    
    # Run gPC in uncertainty group processor 
    if flags[UncertID]:
        UncertOpt.GPC(options=options,benchmarkProblem=benchmarkProblem)
        UncertOpt.evalFunctions(funcs)

    # send data from uncertainty group processor to aero group processor
    if gcomm.rank == uroot:
        gcomm.send(funcs, dest=aroot,tag=11)
    elif gcomm.rank == aroot:
        funcs = gcomm.recv(source=uroot, tag=11)
    # broadcast in aero group processor 
    funcs = gcomm.bcast(funcs,root=aroot)

    # Evaluate functions of constraint in aero group processor
    if flags[aeroID]:
        DVCon.evalFunctions(funcs)
    # broadcast 
    funcs = gcomm.bcast(funcs,root=aroot)

    if MPI.COMM_WORLD.rank == 0:
        DVGeo.writeTecplot(UncertOpt.Optworkdir+'/ffd'+str(UncertOpt.iOpt)+'.dat')

    if MPI.COMM_WORLD.rank == 0:
        print('funcs:')
        print(funcs)
    if MPI.COMM_WORLD.rank == 0:
        with open('out.txt', 'a+', encoding='utf-8') as file:
            print("iopt=",UncertOpt.iOpt,file=file)
            print(p,' ',dim,' ',n_grid,file=file)
            print(UncertOpt.meanres,file = file)
            print(UncertOpt.stdres,file = file)
        with open('outdv.txt', 'a+', encoding='utf-8') as file:
            print("iopt=",UncertOpt.iOpt,file=file)
            print(x,file = file)
        with open('outbenchmark.txt', 'a+', encoding='utf-8') as file:
            print("iopt=",UncertOpt.iOpt,file=file)
            print(UncertOpt.benchmark_funcs,file = file)
    return funcs


def cruiseFuncsSens(x, funcs):

    funcsSens = {}
    
    if flags[UncertID]:
        UncertOpt.evalFunctionsSens(funcsSens,options)

    # send data from uncertainty group processor to aero group processor
    if gcomm.rank == uroot:
        gcomm.send(funcsSens, dest=aroot,tag=11)
    elif gcomm.rank == aroot:
        funcsSens = gcomm.recv(source=uroot, tag=11)
    # broadcast in aero group processor 
    funcsSens = gcomm.bcast(funcsSens,root=aroot)

    # cal sense of constraint in aero group processor
    if flags[aeroID]:
        DVCon.evalFunctionsSens(funcsSens)
    # broadcast in aero group processor
    funcsSens = gcomm.bcast(funcsSens,root=aroot)

    if MPI.COMM_WORLD.rank == 0:
        print('funcsSens:')
        print(funcsSens)

    return funcsSens


def objCon(funcs, printOK):
    # Assemble the objective and any additional constraints:
    funcs["obj"] = funcs[name + '_UQOpt_cd']
    funcs["cl_con_" + name] = funcs[name + '_benchmark_cl'] - mycl
    funcs['cmz_con_'+ name] = funcs[name + '_benchmark_cmz'] - mycm
    if printOK:
        print("funcs in obj:", funcs)
    return funcs


# rst funcs (end)
# ======================================================================
#         Optimization Problem Set-up
# ======================================================================
# rst optprob (beg)
# Create optimization problem
optProb = Optimization("opt", MP.obj, comm=MPI.COMM_WORLD)

# Add objective
optProb.addObj("obj", scale=1e4)

benchmarkProblem.addVariablesPyOpt(optProb)
# Add variables from the Uncertainty Optimization

# UncertOpt.addVariablesPyOpt(optProb)

# Add DVGeo variables
DVGeo.addVariablesPyOpt(optProb)

# Add constraints
if flags[aeroID]:
    DVCon.addConstraintsPyOpt(optProb)
optProb.addCon("cl_con_" + name, lower=-0.001, upper=0.001, scale=1.0)
optProb.addCon('cmz_con_' + name, lower=-0.1, upper=0.0, scale=1.0)

# The MP object needs the 'obj' and 'sens' function for each proc set,
# the optimization problem and what the objcon function is:
MP.setProcSetObjFunc("cruise", cruiseFuncs)
MP.setProcSetSensFunc("cruise", cruiseFuncsSens)
MP.setObjCon(objCon)
MP.setOptProb(optProb)
optProb.printSparsity()
# rst optprob (end)
# rst optimizer
# Set up optimizer
if args.opt == "SLSQP":
    optOptions = {"IFILE": os.path.join(args.output, "SLSQP.out")}
elif args.opt == "SNOPT":
    optOptions = {
        "Major feasibility tolerance": 1e-4,
        "Major optimality tolerance": 1e-4,
        "Difference interval": 1e-3,
        "Major step limit":0.06,
        "Hessian full memory": None,
        "Function precision": 1e-8,
        "Print file": os.path.join(args.output, "SNOPT_print.out"),
        "Summary file": os.path.join(args.output, "SNOPT_summary.out"),
    }
optOptions.update(args.optOptions)
opt = OPT(args.opt, options=optOptions)

# Run Optimization
hotStartfile = "./output/opt_hist1.hst"
sol = opt(optProb, MP.sens, storeHistory=os.path.join(args.output, "opt.hst"),hotStart = hotStartfile )
#sol = opt(optProb, MP.sens, storeHistory=os.path.join(args.output, "opt.hst"))
if MPI.COMM_WORLD.rank == 0:
    print(sol)




