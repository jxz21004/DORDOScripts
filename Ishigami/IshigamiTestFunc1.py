"""
Chenyifu test:
=============================
test for Ishigami function in Static algorithm
no gradient-enhanced
"""

import pygpc
import h5py
import numpy as np
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pygpc.testfunctions import plot_testfunction as plot
from scipy.signal import savgol_filter
import warnings
import seaborn as sns
from scipy.linalg import qr
import pprint
# from mpi4py import MPI
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--MatrixRatio", type=float, default=1.)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--ngrid",type=int,default=None)
parser.add_argument("--GradEnh",type=bool,default=False)
parser.add_argument("--sparse", type=float, default=1.0)
parser.add_argument("--samplemethod",type=str,default="qr")
parser.add_argument("--gpcSolver",type=str,default="Moore-Penrose")
parser.add_argument("--qrratio",type=float,default=1.2)
args = parser.parse_args()

p = args.p
GradEnh = args.GradEnh
sparse = args.sparse
samplemethod = args.samplemethod
qrratio =  args.qrratio
#####################
# Algorithm  Switch #
#####################

algorithm_type = 'Static'

fn_results = 'results/static'
print('==========================================')
print('=  Static version : p='+ str(p) )
print('==========================================')
save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)

###############
# Setup Model #
###############
model = pygpc.testfunctions.Ishigami()

print('plot testfunction Ishigami !')

# plot testfunction : Ishigami
# plot testfunction : Ishigami
p_parameters = OrderedDict()
p_parameters["x1"] = np.linspace(-np.pi, np.pi, 100)
p_parameters["x2"] = np.linspace(-np.pi, np.pi, 100)
p_parameters["x3"] = np.linspace(-np.pi, np.pi, 100)

constants = OrderedDict()
constants["a"] = 7.
constants["b"] = 0.1

print('####################################################')
print('#     Have finished the definition of model!!      #')
print('####################################################')

#################
# Setup Problem #
#################

parameters = OrderedDict()
# parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x1"] = pygpc.Norm(pdf_shape=[0.0, 1.0])
parameters["x2"]  = pygpc.Norm(pdf_shape=[0.0, 1.0])
parameters["x3"]  = pygpc.Norm(pdf_shape=[0.0, 1.0])
parameters["a"] = 7.
parameters["b"] = 0.1

problem = pygpc.Problem(model, parameters)

print('Problem dimension is :',problem.dim)
print('Random parameters : ')
print(problem.parameters_random)
print('####################################################')
print('#    Have finished the definition of problem!!     #')
print('####################################################')


###############
# gPC options #
###############
options = dict()

# setup grid
options["grid"] = pygpc.Random
options["grid_options"] = None

# setup order information
options["order"] = [p]*problem.dim
# options["order_max"] = p*problem.dim
options["order_max"] = p
options["order_max_norm"] = sparse
options["interaction_order"] = 20

# setup solver
# options["solver"] = "LarsLasso"
# options["solver"] = "Moore-Penrose"
options["solver"] = args.gpcSolver

# setup method (only for static algorithm)
options["method"] = "reg"

# setup cpu parallels
options["n_cpu"] = 0

# save format
options["save_session_format"] = save_session_format
options["fn_results"] = fn_results

# specific options
if GradEnh:
    options["gradient_enhanced"] = True
else:
    options["gradient_enhanced"] = False
options["gradient_calculation"] = "External"
options["exGrad_options"] = "TestFunc"
# else:
# options["gradient_enhanced"] = False

options["error_type"] = "nrmsd"
# options["error_type"] = "loocv"
options["eps"] = 0.001
options["backend"] = "omp"        # we now use all CPU cores to calculate the gPC matrices (makes things considerably faster)

# asymptotic sampling weight
# options['asymptotic'] = 'Legendre'
options['asymptotic'] = 'Hermite'
# options['asymptotic']=None

# determine number of basis functions(only for static)
# num_coeffs_sparse = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
#                                         order_glob_max=options["order_max"],
#                                         order_inter_max=options["interaction_order"],
#                                         dim=problem.dim)
                           
# print('determine number of basis functions !')
# print('num_coeffs_sparse :',num_coeffs_sparse)
multi_indices = pygpc.get_multi_indices(order=options["order"], 
                        order_max=options["order_max"], 
                        interaction_order=options["interaction_order"], 
                        order_max_norm=options["order_max_norm"])
n_basis = multi_indices.shape[0]

# # plot the polynomial basis
basis = pygpc.Basis()
basis.init_basis_sgpc(problem=problem,
                      order = options["order"],
                      order_max = options["order_max"],
                      order_max_norm = options["order_max_norm"],
                      interaction_order = options["interaction_order"])
    
# basis.plot_basis(dims=[0,1],fn_plot='fn_plot')



print('n_basis : ',n_basis)
print('basis function number: ',len(basis.b))
print('Check the value of options["order"]: ',options["order"])
print('Check the value of options["order_max"]: ',options["order_max"])
print('Have plot the polynomial basis!')


# generate grid (only for static)
n_grid = args.ngrid
if n_grid is not None:
    if samplemethod=='qr':
        options["matrix_ratio"] = 1.0
        MatrixRatio = 1.0
    else:
        MatrixRatio = n_grid / basis.n_basis
        options["matrix_ratio"] = MatrixRatio
else:
    MatrixRatio = args.MatrixRatio
    options["matrix_ratio"] = MatrixRatio
    n_grid = options["matrix_ratio"] * basis.n_basis
    

print('Matrix ratio is :',options["matrix_ratio"])
print('n_grid : ',n_grid)

# ======================================================================
#         Generate samples
# ======================================================================
if samplemethod=='mc':
    print('Now using sample method "mc"')
    grid = pygpc.Random(parameters_random=problem.parameters_random,
                        n_grid=n_grid,
                        seed=1)
if samplemethod=='lhs':
    print('Now using sample method "lhs"')
    grid = pygpc.LHS(parameters_random=problem.parameters_random,
                        n_grid=n_grid,
                        options='ese')
if samplemethod=='qr':
    print('Now using sample method "qr"')
    n_grid=max(n_grid,10000)

    grid = pygpc.Random(parameters_random=problem.parameters_random,n_grid=n_grid,seed=1)
    # grid = pygpc.LHS(parameters_random=problem.parameters_random,
    #                     n_grid=n_grid,
    #                     options='ese')
    samples = grid.coords
    n_grid=samples.shape[0]
    print("candidate sample scale = ",n_grid)
    asymptotic_weight=np.ones(n_grid)

    if options['asymptotic'] is not None:               
        if options['asymptotic'] == 'Hermite':
            print('Now using asymtotic sampling for Hermite polynominals')
            row_norms = np.linalg.norm(np.array(grid.coords_norm), axis=1) 
            for i in range(n_grid):
                asymptotic_weight[i] = np.exp(-row_norms[i]**2/4)
            options['alpha']=asymptotic_weight
        if options['asymptotic'] == 'Legendre':                            
                print('Now using asymtotic sampling for Legendre polynominals')
                for i in range(n_grid):
                    for j in range(grid.dim):
                        asymptotic_weight[i] *= (abs(1-grid.coords_norm[i][j]**2))**0.25
                options['alpha']=asymptotic_weight
    if options["gradient_enhanced"] is False :
        problem_tem = pygpc.Problem(model, parameters)
        basis_tem = pygpc.Basis()
        basis_tem.init_basis_sgpc(problem=problem_tem,
                order=options["order"],
                order_max=options["order_max"],
                order_max_norm=options["order_max_norm"],
                interaction_order=options["interaction_order"])
        GPC_tem = pygpc.GPC(problem=problem_tem,options=options)
        GPC_tem.basis = basis_tem
        # A is the matrix that should be dealt with qr decomposition
        # Remember to make the form of A be like that during decomposition, the column number needs to be larger than the row number
        # As a result, A should be transposed to A_T before qr decompositon
        A = GPC_tem.create_gpc_matrix(b=basis_tem.b,x=grid.coords_norm)
        # Multiply with weight (initially all the numbers are 1)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] *= asymptotic_weight[i]

        A_T = np.transpose(A)
        # qr decompositon and return QRP
        Q, R, P = qr(A_T, pivoting=True)
        # C = [A[i] for i in P[:n_basis]]
        # make n_grid become P+1
        samples_new = [samples[i] for i in P[:basis_tem.n_basis]]
        # samples_new = samples[P[:basis_tem.n_basis]]# if P is a numpy array, it is the same result as the upper command
        samples = np.array(samples_new)
        n_grid = basis_tem.n_basis
        n_basis = basis_tem.n_basis
        #Update class grid     
        grid = pygpc.Grid(parameters_random=problem.parameters_random,coords=samples)
        grid.coords_norm = grid.get_normalized_coordinates(samples)
    if options["gradient_enhanced"] is True :
        # raise NameError("se-gPC for 'qr' method is not ready for now!")
        # In order not to make large modification to pygpc and pyUncertainty, This part only select sample points with all their gradients, instead of only selecting the simple gradient.  
        # This time normalized samples are used instead of the original ones, but the advantages or disadvantages of doing this are unknown
        problem_tem = pygpc.Problem(model, parameters)
        basis_tem = pygpc.Basis()
        basis_tem.init_basis_sgpc(problem=problem_tem,
                order=options["order"],
                order_max=options["order_max"],
                order_max_norm=options["order_max_norm"],
                interaction_order=options["interaction_order"])
        GPC_tem = pygpc.GPC(problem=problem_tem,options=options)
        GPC_tem.basis = basis_tem
        gradient_idx_B = list(range(0, n_grid))
        # What should be noticed that x of A is grid.coords_norm, and the same as B
        A = GPC_tem.create_gpc_matrix(b=basis_tem.b,x=grid.coords_norm)
        B = GPC_tem.create_gpc_matrix(b=basis_tem.b,x=grid.coords_norm,gradient=True,gradient_idx=gradient_idx_B)
        B = pygpc.misc.ten2mat(B)
        # Multiply with weight (initially all the numbers are 1)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] *= asymptotic_weight[i]
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] *= asymptotic_weight[(i//grid.dim)]
        A_T = np.transpose(A)
        B_T = np.transpose(B)
        C_T = np.concatenate((A_T, B_T), axis=1)
        # qr decompositon and return QRP
        # Originally here should be C_T, but A_T is assumed to be more stable temporarily
        Q, R, P = qr(A_T, pivoting=True)
        samples_new=[]
        for i in P[:min(basis_tem.n_basis,int((np.ceil(basis_tem.n_basis/(grid.dim+1)))*qrratio))]:
            if (i<n_grid):
                samples_new.append(samples[i])
            else:
                samples_new.append(samples[(i-n_grid)//grid.dim])
        # Remove duplicate data
        print('================P=================')
        pprint.pprint(P[:2*basis_tem.n_basis])
        pprint.pprint(R[:basis_tem.n_basis, :basis_tem.n_basis].diagonal())
        print('==================================')
        samples = np.array(samples_new)   
        samples = np.append(samples, [np.zeros(samples.shape[1])+1e-7],axis=0) 
        print('sample.shape[1]=',samples.shape[1],'grid.dim=',grid.dim)
        samples = np.unique(samples, axis=0)
        # This time n_grid might be smaller than n_basis
        n_grid = samples.shape[0]
        n_basis = basis_tem.n_basis
        # Update class grid     
        grid = pygpc.Grid(parameters_random=problem.parameters_random,coords=samples)
        grid.coords_norm = grid.get_normalized_coordinates(samples)

print('The shape of grid is :',grid.coords.shape)
print('save grid data !')
np.savetxt('grid.dat',grid.coords)
np.savetxt('normal_grid.dat', grid.coords_norm)
pprint.pprint(grid.coords_norm)
# define external gradient solver
Ishigami_Grad = pygpc.Ishigami_Grad(parameters)
# define algorithm
if GradEnh:
    algorithm = pygpc.Static(problem=problem, options=options, grid=grid, exGrad=None, exGradSolver=Ishigami_Grad)
else:
    algorithm = pygpc.Static(problem=problem, options=options, grid=grid)


print('#################################################################')
print('#    Have finished the settings of calculation parameters!!     #')
print('#################################################################')

if options["gradient_enhanced"] is True:
    print('--> Gradient Enhanced')

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC session
session, coeffs, results, eps = session.run()

print('The shape of grid is :',session.grid.coords.shape)
# print(session.grid.coords)
print('The shape of gpc matrix is :',session.gpc[0].gpc_matrix.shape)

print('The shape of coeffs is :',coeffs.shape)
# print(coeffs)
np.savetxt('coeffs.dat',coeffs)

print('The shape of outputs is :',results.shape)
np.savetxt('results.dat',results)

meanres, stdres = pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                                output_idx=None,
                                calc_sobol=True,
                                calc_global_sens=False,
                                calc_pdf=False,
                                algorithm = "standard",
                                n_samples=1e5)
for i in range(1):
    print(i,meanres[0,i],stdres[0,i]*stdres[0,i])
print("done!\n")

print('#################################################################')
print('#            Have finished the run of gPC models!!              #')
print('#################################################################')


################################################
#            Post-processing data              #
################################################

# Post-process gPC and add results to .hdf5 file
pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             n_samples=1e4)

# Validate gPC vs original model function
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=["x1", "x2"],
                        n_grid=[51, 51],
                        output_idx=0,
                        fn_out=session.fn_results + '_val',
                        n_cpu=session.n_cpu)

# Validate gPC vs original model function (Monte Carlo)
nrmsd, y_orig, y_gpc = pygpc.validate_gpc_mc(session=session,
                                             coeffs=coeffs,
                                             n_samples=int(1e6),
                                             output_idx=0,
                                             n_cpu=session.n_cpu,
                                             fn_out=session.fn_results + '_mc')
ydelta=(y_orig-y_gpc)
maxindex=np.argmax(abs(np.array(ydelta)))
ymean=np.mean(np.array(y_orig))
ymean_gpc=np.mean(np.array(y_gpc))
ystd=np.var(np.array(y_orig))**0.5
ystd_gpc=np.var(np.array(y_gpc))**0.5
print('maxdelta=',max(abs(ydelta)),max(abs(ydelta))/(np.mean(np.array(y_orig))))
print('the mean of y_orig=',ymean,'y_gpc',ymean_gpc)
print('the std of y_orig=',ystd,'y_gpc',ystd_gpc)
print('the point orig=',y_orig[maxindex],'y_gpc=',y_gpc[maxindex])
# Validate gPC vs original model function (Only samples)
error = pygpc.nrmsd_gpc_vs_origin(session=session, coeffs=coeffs, results=results)

# print('y_orig',y_orig)
# print('y_gpc',y_gpc)

print('+---------------------------------------------------+')
# print('| Maximum NRMSD (gpc vs original, Monte Carlo): {:.2}'.format(max(nrmsd)))
# print('| Maximum NRMSD (gpc vs original, Only samples): {:.2}'.format(max(error)))
print('| Maximum NRMSD (gpc vs original, Monte Carlo):',nrmsd)
print('| Maximum NRMSD (gpc vs original, Only samples):',error)
print('+---------------------------------------------------+')

################################################
#       Extract Post-processing data           #
################################################

print('Extract Post-processing data')
print('y_orig:',y_orig.shape)
print('y_gpc:',y_gpc.shape)


np.savetxt(session.fn_results+'_validation_orig.dat', y_orig)
np.savetxt(session.fn_results+'_validation_gpc.dat', y_gpc)


print("done!\n")

################################################
#    Write results into a summaray file        #
################################################
ishi_mean=3.5
ishi_sq=13.844587940719254
# ishi_1=ishi_mean
# ishi_2=ishi_sq**0.5
ishi_1=ymean
ishi_2=ystd
# ishi_1=3.033858338
# ishi_2=2.612169737
ishi_3=0.
ishi_4=3.5072
filepath = '../'
f1 = open(filepath+'error_summary.dat','a')
f1.write(samplemethod)
# f1.write('%5i\t%5i\t%20.10f\t%20.10f\t%20.10f\t%20.10f\t%20.10f\n'%(p,n_grid,eps,nrmsd,np.abs((meanres[0,0]-3.5))/3.5,np.abs((stdres[0,i]*stdres[0,i]-13.844587940719254))/13.844587940719254,np.abs((meanres[0,0]-3.5))/3.5+np.abs((stdres[0,i]*stdres[0,i]-13.844587940719254))/13.844587940719254))
f1.write('%5d\t%20.2f%20.2f\n'%(GradEnh,sparse,qrratio))
f1.write('%5i\t%5i\t%20.10f\t%20.10f\t%20.10f\t%20.10f\t%20.10f\n'%(p,n_grid,eps,nrmsd,np.abs((ymean_gpc-ishi_1))/ishi_1,np.abs((ystd_gpc-ishi_2))/ishi_2,np.abs((ymean_gpc-ishi_1))/ishi_1+np.abs((ystd_gpc-ishi_2))/ishi_2))
f1.write('%20.10f\t\n'%np.abs(((ymean_gpc)**2+ystd_gpc**2)/(ishi_1**2+ishi_2**2)-1))
f1.close()
f2 = open(filepath+'error_sum_detail.dat','a')
f2.write(samplemethod)
f2.write('%5d\t%20.2f%20.2f\n'%(GradEnh,sparse,qrratio))
f2.write('%5i\t%5i\t%20.10f\t%20.10f\t%20.10f\t%20.10f\t%20.10f\n'%(p,n_grid,eps,nrmsd,np.abs((ymean_gpc-ishi_1))/ishi_1,np.abs((ystd_gpc-ishi_2))/ishi_2,np.abs((ymean_gpc-ishi_1))/ishi_1+np.abs((ystd_gpc-ishi_2))/ishi_2))
f2.write('%20.10f\t\n'%np.abs(((ymean_gpc)**2+ystd_gpc**2)/(ishi_1**2+ishi_2**2)-1))
print('maxdelta=',max(abs(ydelta)),max(abs(ydelta))/(np.mean(np.array(y_orig))),file=f2)
print('the mean of y_orig=',ymean,'y_gpc',ymean_gpc,file=f2)
print('the std of y_orig=',ystd,'y_gpc',ystd_gpc,file=f2)
print('the point orig=',y_orig[maxindex],'y_gpc=',y_gpc[maxindex],file=f2)
# f2 = open(filepath+'nrmsd-p.dat','a')
# if MatrixRatio == 1:
#     f2.write('%i\t%12.7f\t'%(p,nrmsd))
# else:
#     f2.write('%12.7f\n'%(nrmsd))

f2.close()