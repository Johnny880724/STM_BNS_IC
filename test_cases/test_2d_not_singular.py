# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:56:24 2020

@author: Johnny Tsao
"""
import mesh_helper_functions_3d as mhf3d
import mesh_helper_functions as mhf
import test_cases_3d as tc
import jacobi_newtonian_solver_n as jns
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os

cwd = os.getcwd()
plt.close("all")
    
single_test = False
if(single_test):
    N_grid = 60
    u_init, (xmesh,ymesh), h = setup_grid(N_grid)
    r0 = 1.0
    r = mhf.XYtoR(xmesh,ymesh)
    rho = r**2 - r0**2
    phi = r**2 - r0**2 + 0.1
    param = 0.5
    theory = ymesh + param * np.sin(xmesh)
    boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
    rhs = 2*(ymesh + param * xmesh* np.cos(xmesh)) - param*(r**2 - r0**2)*np.sin(xmesh)
    frame = mhf.get_frame(phi)
    frame_full =mhf.get_frame(rho)
#    plt.matshow(theory*frame)
    ux, uy = mhf.grad(theory,h,h)
    rhs_theory = mhf.div(rho*ux,rho*uy,h,h)
    
    #test
#    mhf3d.plot2d_compare(rhs,rhs_theory,frame)
    # maximum iteration number for the source term method
    maxIter = 100
    # the total iteration number N for Jacobi solver = it_multi * N_grid**2
    it_multi = 10
    u_extpl_result = jns.stm_coef_Neumann(u_init, (xmesh,ymesh),phi,rhs,rho,\
                                       theory*frame, boundary, maxIter)
    
    
grid_test = True
if(grid_test):
    # grid_array = [int(2**(i*0.05)) for i in range(100,160)]
    # grid_array = [int(2**(i*0.05)) for i in range(100,110)]
    grid_array = list(dict.fromkeys([2*int(2**(i*0.05)) for i in range(80,140)])) 
    # grid_array = [32,64]
    L2Dif_array_extpl = []
    L2Dif_array = []
    vepc_result_array = []
    theory_array = []
    it_array = []
    for N_grid in grid_array:
        u_init, (xmesh,ymesh), h = jns.setup_grid(N_grid)
        # r0 = 0.8
        r0 = np.sqrt(np.e/4)
        r = mhf.XYtoR(xmesh,ymesh)
        ecc = 1.2
        rp = mhf.XYtoR(ecc*xmesh,ymesh)
        dh_coef = 0.10
        rho = -(rp**2 - r0**2)
        rhoc = np.max(np.abs(rho))
        phi = rho - dh_coef
        coef = np.exp(r**2)
        param = 0.5
        theory = (1 + np.sin(xmesh)*np.cos(ymesh))
        boundary = (ecc**2*xmesh*np.cos(xmesh)*np.cos(ymesh) - ymesh*np.sin(xmesh)*np.sin(ymesh)) \
                    / (mhf.XYtoR(ecc**2*xmesh,ymesh) + mhf.singular_null)
        rhs = 2*np.exp(r**2)*(xmesh*np.cos(xmesh)*np.cos(ymesh) - ymesh*np.sin(xmesh)*np.sin(ymesh) - np.sin(xmesh)*np.cos(ymesh)) 
        frame = mhf.get_frame_n(phi)
        frame_full =mhf.get_frame_n(rho)
        ux, uy = mhf.grad(theory,h,h)
        rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
        # test rhs
        # mhf3d.plot2d_compare(rhs,rhs_theory,frame_full,"rhs")
        # test boundary
        # mhf3d.plot2d_compare(boundary,mhf.grad_n_n(theory,phi,h,h),jns.get_N1(phi),"boundary")
        # maximum iteration number for the source term method
        maxIter = 1000
        # the total iteration number N for Jacobi solver = it_multi * N_grid**2
        it_multi = 10
        eta = 1.0e-4
        u_extpl_result, it = jns.stm_coef_Neumann(u_init, (xmesh,ymesh),phi,rho,rhs,coef,\
                                       theory*frame, boundary, maxIter,eta)
        
#        mhf3d.plot2d_compare(u_extpl_result,theory,frame)
        # mhf3d.plot2d_compare_zero(u_extpl_result,theory,frame_full)
        
        maxDif_extpl, L2Dif_extpl = mhf.get_error_N(u_extpl_result, theory, frame_full,(False,False))
        L2_rel_dif_extpl = L2Dif_extpl / np.sqrt(np.mean((theory)**2))
        print("grid size " + str(N_grid) + ": L2 extpl error " ,L2_rel_dif_extpl)
        
        maxDif, L2Dif = mhf.get_error_N(u_extpl_result, theory, frame)
        L2_rel_dif_extpl = L2Dif_extpl / np.sqrt(np.mean((theory)**2))
        L2Dif_array_extpl.append(L2_rel_dif_extpl)
        L2Dif_array.append(L2Dif)
        vepc_result_array.append(u_extpl_result)
        theory_array.append(theory)
        it_array.append(it)
        # print(L2_rel_dif)
        
    plot_grid_L2dif = True
    if(plot_grid_L2dif):
        plt.close("all")
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-1,1)) 
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(grid_array,  L2Dif_array_extpl, label = "after extrapolation")
        ax.plot(grid_array,  L2Dif_array_extpl,"b.")
        ax.set_yscale("log",basey = 10)
        ax.set_xscale("log",basex = 2)
        # plt.plot(np.log(grid_array),  np.log(L2Dif_array_extpl), label = "after extrapolation")
        # plt.plot(np.log(grid_array),  np.log(L2Dif_array_extpl),"b.")
        plt.plot(grid_array,  L2Dif_array, label = "before extrapolation")
        plt.plot(grid_array,  L2Dif_array,"r.")
        ax.set_xlabel("grid size")
        ax.set_ylabel("L2 relative error")
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        from decimal import Decimal
        ax.set_title("test case 2d convergence")
        plt.legend()
    
    
    import h5py as h5
    import os
    cwd = os.getcwd()
    store_file = True
    if(store_file):
        store_data_file = h5.File(cwd+"\\2d_basic\\test_case_1_rlx_dh10"+ "_noQ.h5","w")
        store_data_file.create_dataset("Ngrid",data = grid_array)
        store_data_file.create_dataset("L2",data = L2Dif_array)
        store_data_file.create_dataset("L2extpl",data = L2Dif_array_extpl)
        store_data_file.create_dataset("sigma_r0_param_eta",data = np.array([dh_coef,r0,param,eta]))
        for i in range(len(vepc_result_array)):
            store_data_file.create_dataset("vepc" + str(i),data = vepc_result_array[i])
            store_data_file.create_dataset("theory" + str(i),data = theory_array[i])
        store_data_file.create_dataset("it",data = it_array)
        store_data_file.close()