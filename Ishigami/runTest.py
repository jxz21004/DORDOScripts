import os
import sys
import numpy as np

p = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
n_grid =10000
samplemethod = 'lhs'
gradenhlist=[True,False]
sparse=[1.0]
qrratiolist=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
filepath='./'
print('####################################################################')
print('#                Run Test function PCE approximation               #')
print('####################################################################')

orgpath=os.getcwd()
filepath = './'
f = open(filepath+'error_summary.dat','w')

f.write('TITLE     = ""\n')
f.write('VARIABLES = "Order"\n') 
f.write('"Samples number"\n') 
f.write('"loocv"\n') 
f.write('"nrmsd"\n')
f.write('"mean"\n')
f.write('"std"\n')
f.write('"mean+std"\n')
f.write('"Soom"\n')
f.write('ZONE T="Ishigami"\n')
# f.write('STRANDID=0, SOLUTIONTIME=0\n') 
# f.write('I='+str(len(p))+', J='+str(len(n_grid))+' ,K=1, ZONETYPE=Ordered\n') 
# f.write('DATAPACKING=POINT\n') 
# f.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n') 
f.close()
os.system('. /work/home/ac8ibz8v5y/packages/setenv1.sh')
if samplemethod=='qr':
    for i in p:
        for j in gradenhlist:
            if j==True:
                for k in qrratiolist:
                    for l in sparse:
                        os.chdir(orgpath)
                        tempath='p'+str(i)+'grad'+str(j)+'r'+str(k)+'sparse'+str(l)
                        if(not(os.path.exists(tempath))):
                            os.mkdir(tempath)
                        os.chdir(tempath)
                        print('--->  p :', i)
                        # os.system('cp ../IshigamiTestFunc1.py IshigamiTestFunc1.py')
                        os.system('python ../IshigamiTestFunc1.py --p {}  --samplemethod {} --ngrid {} --GradEnh {} --sparse {} --qrratio {} 2>&1|tee m.log'.format(i,samplemethod,n_grid,j,l,k))
                        os.chdir(os.path.pardir)
            else:
                    for l in sparse:
                        os.chdir(orgpath)
                        tempath='p'+str(i)+'grad'+str(j)+'r'+str(k)+'sparse'+str(l)
                        if(not(os.path.exists(tempath))):
                            os.mkdir(tempath)
                        os.chdir(tempath)
                        print('--->  p :', i)
                        # os.system('cp ../IshigamiTestFunc1.py IshigamiTestFunc1.py')
                        os.system('python ../IshigamiTestFunc1.py --p {}  --samplemethod {} --ngrid {} --sparse {} --qrratio {} 2>&1|tee m.log'.format(i,samplemethod,n_grid,l,k))
                        os.chdir(os.path.pardir)
if samplemethod=='lhs':
    for i in p:
        for j in gradenhlist:
            if j==True:
                for k in qrratiolist:
                    for l in sparse:
                        k1=k/4+0.01
                        os.chdir(orgpath)
                        tempath='p'+str(i)+'grad'+str(j)+'r'+str(k1)+'sparse'+str(l)
                        if(not(os.path.exists(tempath))):
                            os.mkdir(tempath)
                        os.chdir(tempath)
                        print('--->  p :', i)
                        # os.system('cp ../IshigamiTestFunc1.py IshigamiTestFunc1.py')
                        os.system('python ../IshigamiTestFunc1.py --p {}  --samplemethod {} --GradEnh {} --sparse {} --MatrixRatio {} --qrratio {} 2>&1|tee m.log'.format(i,samplemethod,j,l,k1,k1))
                        os.chdir(os.path.pardir)
                for k in qrratiolist:
                    for l in sparse:
                        os.chdir(orgpath)
                        tempath='p'+str(i)+'grad'+str(j)+'r'+str(k)+'sparse'+str(l)
                        if(not(os.path.exists(tempath))):
                            os.mkdir(tempath)
                        os.chdir(tempath)
                        print('--->  p :', i)
                        # os.system('cp ../IshigamiTestFunc1.py IshigamiTestFunc1.py')
                        os.system('python ../IshigamiTestFunc1.py --p {}  --samplemethod {} --GradEnh {} --sparse {} --MatrixRatio {} --qrratio {} 2>&1|tee m.log'.format(i,samplemethod,j,l,k,k))
                        os.chdir(os.path.pardir)
            else:
                for k in qrratiolist:
                    for l in sparse:
                        os.chdir(orgpath)
                        tempath='p'+str(i)+'grad'+str(j)+'r'+str(k)+'sparse'+str(l)
                        if(not(os.path.exists(tempath))):
                            os.mkdir(tempath)
                        os.chdir(tempath)
                        print('--->  p :', i)
                        # os.system('cp ../IshigamiTestFunc1.py IshigamiTestFunc1.py')
                        os.system('python ../IshigamiTestFunc1.py --p {}  --samplemethod {} --sparse {} --MatrixRatio {} --qrratio {} 2>&1|tee m.log'.format(i,samplemethod,l,k,k))
                        os.chdir(os.path.pardir)