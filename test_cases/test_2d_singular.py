# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:56:24 2020

@author: Johnny Tsao
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
import mesh_helper_functions_3d as mhf3d
import mesh_helper_functions as mhf
import jacobi_newtonian_solver_n as jns
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# cwd = "/Users/btsao/Dropbox/Numerical_Relativity/2019_SPIN/BNS_IC_SOLVER/newtonian3d_solver"
cwd = os.getcwd()
plt.close("all")
    
# generating data for plot 2-1,2
const_sigma = False
if(const_sigma):
    # grid_array = [int(2**(i*0.05)) for i in range(100,160)]
    grid_array = list(dict.fromkeys([2*int(2**(i*0.05)) for i in range(80,140)])) 
    # grid_array = [int(2**(i*0.05)) for i in [100, 135, 160]]
    # grid_array = [123]
    L2Dif_array_extpl = []
    L2Dif_array = []
    vepc_result_array = []
    theory_array = []
    it_array = []
    sigma_array = []
    for N_grid in grid_array:
    # for N_grid in [32]:
        u_init, (xmesh,ymesh), h = jns.setup_grid(N_grid)
        r0 = np.sqrt(np.e/4)
        r = mhf.XYtoR(xmesh,ymesh)
        
        
        power_dict = {"1":2.0, "2":1.0, "3":0.5}
        test_case = "3"
        power = power_dict[test_case]
        
        rlx = 0.1
        sigma = 0.10
        
        ## test case 1 -- rho = r^2
        if(test_case == "1"):
            rho = -(r**2 - r0**2)
            rhoc = r0**2
            coef = np.copy(rho)
            phi = rho - sigma * rhoc
            param = 1.0
            theory = ymesh + param * np.sin(xmesh)
            boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
            rhs = -(2*(ymesh + param * xmesh* np.cos(xmesh)) - param*(r**2 - r0**2)*np.sin(xmesh))
                
            frame = mhf.get_frame_n(phi)
            frame_full =mhf.get_frame_n(rho)
            ux, uy = mhf.grad(theory,h,h)
            rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
            
            # plt.matshow((1.0/rho+mhf3d.singular_null)*frame)
        ## test case 2 -- rho = r
        elif(test_case == "2"):
            rho = -(r - r0)
            rhoc = r0
            coef = np.copy(rho)
            phi = rho - sigma * rhoc
            param = 1.0
            theory = ymesh + param * np.sin(xmesh)
            boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
            rhs = -((ymesh + param * xmesh* np.cos(xmesh)) - param*r*(r - r0)*np.sin(xmesh))/ (r + mhf.singular_null)
                
            frame = mhf.get_frame_n(phi)
            frame_full = mhf.get_frame_n(rho)
            ux, uy = mhf.grad(theory,h,h)
            rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
        
        ## test case 3 -- rho = r^0.5
        elif(test_case == "3"):
            rho = -(r**0.5 - r0**0.5)
            rhoc = r0
            coef = np.copy(rho)
            phi = rho - sigma * rhoc
            param = 1.0
            theory = ymesh + param * np.sin(xmesh)
            boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
            rhs = -(ymesh + param * xmesh* np.cos(xmesh))/ (2*r**1.5 + mhf.singular_null) - param*rho*np.sin(xmesh)
                
            frame = mhf.get_frame_n(phi)
            frame_full =mhf.get_frame_n(rho)
            ux, uy = mhf.grad(theory,h,h)
            rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
        
        else:
            print("invalid test case")
            break
        
        test = [False, False]
        if(test[0]):
            mhf3d.plot2d_compare(rhs,rhs_theory,frame_full,"rhs")
            
        # test boundary
        if(test[1]):
            mhf3d.plot2d_compare(boundary,mhf.grad_n_n(theory,phi,h,h),jns.get_N1(phi),"boundary")
        # maximum iteration number for the source term method
        maxIter = 1000
        # the total iteration number N for Jacobi solver = it_multi * N_grid**2
        it_multi = 10
        eta = 1.0e-4
        u_extpl_result, it = jns.stm_coef_Neumann(u_init, (xmesh,ymesh),phi,rho,rhs,coef,\
                                        theory*frame, boundary, maxIter,eta,rlx)
        
#        mhf3d.plot2d_compare(u_extpl_result,theory,frame)
        # mhf3d.plot2d_compare_zero(u_extpl_result,theory,frame_full)
        
        maxDif_extpl, L2Dif_extpl = mhf.get_error_N(u_extpl_result, theory, frame_full,(True,False))
        maxDif, L2Dif = mhf.get_error_N(u_extpl_result, theory, frame,(False,False))
        L2_rel_dif_extpl = L2Dif_extpl / np.sqrt(np.mean((theory)**2))
        L2Dif_array_extpl.append(L2_rel_dif_extpl)
        L2Dif_array.append(L2Dif)
        vepc_result_array.append(u_extpl_result)
        theory_array.append(theory)
        it_array.append(it)
        print("grid size " + str(N_grid) + ": L2 extpl error " ,L2_rel_dif_extpl)
        
        
        
    plot_grid_L2dif = True
    if(plot_grid_L2dif):
        # plt.close("all")
        import matplotlib
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
        ax.set_yticks([1e-3,5e-4,1e-4])
        ax.set_xticks([50,100,150,200])
        ax.set_xlabel("grid size")
        ax.set_ylabel("L2 relative error")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        from decimal import Decimal
        ax.set_title("test case 2d log-log plot")
        plt.legend()
    
    
    import h5py as h5
    import os
    cwd = os.getcwd()
    store_file = False
    if(store_file):
        store_data_file = h5.File(cwd+"\\2d_singular\\const_sigma\\test_case_"+test_case+"_rlx_010_noQ.h5","w")
        store_data_file.create_dataset("Ngrid",data = grid_array)
        store_data_file.create_dataset("L2",data = L2Dif_array)
        store_data_file.create_dataset("L2extpl",data = L2Dif_array_extpl)
        store_data_file.create_dataset("sigma_r0_param_eta",data = np.array([sigma,r0,param,eta]))
        for i in range(len(vepc_result_array)):
            store_data_file.create_dataset("vepc" + str(i),data = vepc_result_array[i])
            store_data_file.create_dataset("theory" + str(i),data = theory_array[i])
        store_data_file.create_dataset("it",data = it_array)
        store_data_file.create_dataset("sigma",data = sigma_array)
        store_data_file.close()
    

# generating data for plot 2-3
vary_sigma = False
if(vary_sigma):
    # grid_array = [int(2**(i*0.05)) for i in range(100,160)]
    grid_array = list(dict.fromkeys([2*int(2**(i*0.05)) for i in range(80,140)])) 
    # grid_array = [int(2**(i*0.05)) for i in [100, 135, 160]]
    # grid_array = [123]
    L2Dif_array_extpl = []
    L2Dif_array = []
    vepc_result_array = []
    theory_array = []
    it_array = []
    sigma_array = []
    for N_grid in grid_array:
        u_init, (xmesh,ymesh), h = jns.setup_grid(N_grid)
        r0 = np.sqrt(np.e/4)
        r = mhf.XYtoR(xmesh,ymesh)
        
        # num_grid_dr = 0.933
        num_grid_dr = 5.0
        dr_by_r0 = num_grid_dr * h / r0
        
        power_dict = {"1":2.0, "2":1.0, "3":0.5}
        test_case = "2"
        power = power_dict[test_case]
        
        rlx = 0.1
        sigma = 1 - (1 - dr_by_r0)**power
        sigma_array.append(sigma)
        # sigma = 0.05
        test = [False, False]
        # ecc = 1.2
        # rp = mhf.XYtoR(ecc*xmesh,ymesh)
        # dh_coef = 0.08
        # rho = -(rp**2 - r0**2)
        # rhoc = np.max(np.abs(rho))
        # phi = rho - dh_coef
        # coef = rho
        # param = 0.5
        # theory = (1 + np.sin(xmesh)*np.cos(ymesh))
        # boundary = (ecc**2*xmesh*np.cos(xmesh)*np.cos(ymesh) - ymesh*np.sin(xmesh)*np.sin(ymesh)) \
        #             / (mhf.XYtoR(ecc**2*xmesh,ymesh) + mhf.singular_null)
        # rhs = -2*ecc**2*xmesh*np.cos(xmesh)*np.cos(ymesh) + \
        #     2*np.sin(xmesh)*((rp**2-r0**2)*np.cos(ymesh)+ymesh*np.sin(ymesh)) 
        
        
        ## test case 1 -- rho = r^2
        if(test_case == "1"):
            rho = -(r**2 - r0**2)
            rhoc = r0**2
            coef = np.copy(rho)
            phi = rho - sigma * rhoc
            param = 1.0
            theory = ymesh + param * np.sin(xmesh)
            boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
            rhs = -(2*(ymesh + param * xmesh* np.cos(xmesh)) - param*(r**2 - r0**2)*np.sin(xmesh))
                
            frame = mhf.get_frame_n(phi)
            frame_full =mhf.get_frame_n(rho)
            ux, uy = mhf.grad(theory,h,h)
            rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
            
            # plt.matshow((1.0/rho+mhf3d.singular_null)*frame)
        ## test case 2 -- rho = r
        elif(test_case == "2"):
            rho = -(r - r0)
            rhoc = r0
            coef = np.copy(rho)
            phi = rho - sigma * rhoc
            param = 1.0
            theory = ymesh + param * np.sin(xmesh)
            boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
            rhs = -((ymesh + param * xmesh* np.cos(xmesh)) - param*r*(r - r0)*np.sin(xmesh))/ (r + mhf.singular_null)
                
            frame = mhf.get_frame_n(phi)
            frame_full = mhf.get_frame_n(rho)
            ux, uy = mhf.grad(theory,h,h)
            rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
        
        ## test case 3 -- rho = r^0.5
        elif(test_case == "3"):
            rho = -(r**0.5 - r0**0.5)
            rhoc = r0
            coef = np.copy(rho)
            phi = rho - sigma * rhoc
            param = 1.0
            theory = ymesh + param * np.sin(xmesh)
            boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
            rhs = -(ymesh + param * xmesh* np.cos(xmesh))/ (2*r**1.5 + mhf.singular_null) - param*rho*np.sin(xmesh)
                
            frame = mhf.get_frame_n(phi)
            frame_full =mhf.get_frame_n(rho)
            ux, uy = mhf.grad(theory,h,h)
            rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
        
        else:
            print("invalid test case")
            break
        # plt.matshow(frame_full - frame)
        # test rhs
        if(test[0]):
            mhf3d.plot2d_compare(rhs,rhs_theory,frame_full,"rhs")
            
        # test boundary
        if(test[1]):
            mhf3d.plot2d_compare(boundary,mhf.grad_n_n(theory,phi,h,h),jns.get_N1(phi),"boundary")
        # maximum iteration number for the source term method
        maxIter = 1000
        # the total iteration number N for Jacobi solver = it_multi * N_grid**2
        it_multi = 10
        eta = 1.0e-4
        u_extpl_result, it = jns.stm_coef_Neumann(u_init, (xmesh,ymesh),phi,rho,rhs,coef,\
                                        theory*frame, boundary, maxIter,eta,rlx)
        
#        mhf3d.plot2d_compare(u_extpl_result,theory,frame)
        # mhf3d.plot2d_compare_zero(u_extpl_result,theory,frame_full)
        
        maxDif_extpl, L2Dif_extpl = mhf.get_error_N(u_extpl_result, theory, frame_full,(False,False))
        maxDif, L2Dif = mhf.get_error_N(u_extpl_result, theory, frame,(False,False))
        L2_rel_dif_extpl = L2Dif_extpl / np.sqrt(np.mean((theory)**2))
        L2Dif_array_extpl.append(L2_rel_dif_extpl)
        L2Dif_array.append(L2Dif)
        vepc_result_array.append(u_extpl_result)
        theory_array.append(theory)
        it_array.append(it)
        print("grid size " + str(N_grid) + ": L2 extpl error " ,L2_rel_dif_extpl)
        
        
    plot_grid_L2dif = True
    if(plot_grid_L2dif):
        # plt.close("all")
        import matplotlib
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
        ax.set_yticks([1e-3,5e-4,1e-4])
        ax.set_xticks([50,100,150,200])
        ax.set_xlabel("grid size")
        ax.set_ylabel("L2 relative error")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        from decimal import Decimal
        ax.set_title("test case 2d log-log plot")
        plt.legend()
    
    
    import h5py as h5
    import os
    cwd = os.getcwd()
    store_file = False
    if(store_file):
        store_data_file = h5.File(cwd+"\\2d_singular\\vary_sigma\\test_case_"+test_case+"_rlx_varN"+"{:1}".format(int(num_grid_dr))+ "_noQ.h5","w")
        store_data_file.create_dataset("Ngrid",data = grid_array)
        store_data_file.create_dataset("L2",data = L2Dif_array)
        store_data_file.create_dataset("L2extpl",data = L2Dif_array_extpl)
        store_data_file.create_dataset("sigma_r0_param_eta",data = np.array([sigma,r0,param,eta]))
        for i in range(len(vepc_result_array)):
            store_data_file.create_dataset("vepc" + str(i),data = vepc_result_array[i])
            store_data_file.create_dataset("theory" + str(i),data = theory_array[i])
        store_data_file.create_dataset("it",data = it_array)
        store_data_file.create_dataset("sigma",data = sigma_array)
        store_data_file.close()
        
sigma_test = True
if(sigma_test):
    
    
    grid_array = [32,64,128]
    
    for N_grid in grid_array:
        # num_grid_dr_array =[2.0,1.0]
        num_grid_dr_array = [10**(0.05*idx-2.0) for idx in range(60)]
        L2Dif_array_extpl = []
        L2Dif_array = []
        vepc_result_array = []
        theory_array = []
        it_array = []
        sigma_array = []
        for num_grid_dr in num_grid_dr_array:
            u_init, (xmesh,ymesh), h = jns.setup_grid(N_grid)
            r0 = np.sqrt(np.e/4)
            r = mhf.XYtoR(xmesh,ymesh)
            
            # num_grid_dr = 0.933
            # num_grid_dr = 1.0
            dr_by_r0 = num_grid_dr * h / r0
            
            power_dict = {"1":2.0, "2":1.0, "3":0.5}
            test_case = "2"
            power = power_dict[test_case]
            
            rlx = 0.1
            sigma = 1 - (1 - dr_by_r0)**power
            sigma_array.append(sigma)
            test = [False, False]
            
            ## test case 1 -- rho = r^2
            if(test_case == "1"):
                rho = -(r**2 - r0**2)
                rhoc = r0**2
                coef = np.copy(rho)
                phi = rho - sigma * rhoc
                param = 1.0
                theory = ymesh + param * np.sin(xmesh)
                boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
                rhs = -(2*(ymesh + param * xmesh* np.cos(xmesh)) - param*(r**2 - r0**2)*np.sin(xmesh))
                    
                frame = mhf.get_frame_n(phi)
                frame_full =mhf.get_frame_n(rho)
                ux, uy = mhf.grad(theory,h,h)
                rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
                
                # plt.matshow((1.0/rho+mhf3d.singular_null)*frame)
            ## test case 2 -- rho = r
            elif(test_case == "2"):
                rho = -(r - r0)
                rhoc = r0
                coef = np.copy(rho)
                phi = rho - sigma * rhoc
                param = 1.0
                theory = ymesh + param * np.sin(xmesh)
                boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
                rhs = -((ymesh + param * xmesh* np.cos(xmesh)) - param*r*(r - r0)*np.sin(xmesh))/ (r + mhf.singular_null)
                    
                frame = mhf.get_frame_n(phi)
                frame_full = mhf.get_frame_n(rho)
                ux, uy = mhf.grad(theory,h,h)
                rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
            
            ## test case 3 -- rho = r^0.5
            elif(test_case == "3"):
                rho = -(r**0.5 - r0**0.5)
                rhoc = r0
                coef = np.copy(rho)
                phi = rho - sigma * rhoc
                param = 1.0
                theory = ymesh + param * np.sin(xmesh)
                boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
                rhs = -(ymesh + param * xmesh* np.cos(xmesh))/ (2*r**1.5 + mhf.singular_null) - param*rho*np.sin(xmesh)
                    
                frame = mhf.get_frame_n(phi)
                frame_full =mhf.get_frame_n(rho)
                ux, uy = mhf.grad(theory,h,h)
                rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
            
            else:
                print("invalid test case")
                break
            # plt.matshow(frame_full - frame)
            # test rhs
            if(test[0]):
                mhf3d.plot2d_compare(rhs,rhs_theory,frame_full,"rhs")
                
            # test boundary
            if(test[1]):
                mhf3d.plot2d_compare(boundary,mhf.grad_n_n(theory,phi,h,h),jns.get_N1(phi),"boundary")
            # maximum iteration number for the source term method
            maxIter = 200
            # the total iteration number N for Jacobi solver = it_multi * N_grid**2
            it_multi = 10
            eta = 1.0e-4
            u_extpl_result, it = jns.stm_coef_Neumann(u_init, (xmesh,ymesh),phi,rho,rhs,coef,\
                                            theory*frame, boundary, maxIter,eta,rlx)
            
    #        mhf3d.plot2d_compare(u_extpl_result,theory,frame)
            # mhf3d.plot2d_compare_zero(u_extpl_result,theory,frame_full)
            
            maxDif_extpl, L2Dif_extpl = mhf.get_error_N(u_extpl_result, theory, frame_full,(False,False))
            maxDif, L2Dif = mhf.get_error_N(u_extpl_result, theory, frame,(False,False))
            L2_rel_dif_extpl = L2Dif_extpl / np.sqrt(np.mean((theory)**2))
            L2Dif_array_extpl.append(L2_rel_dif_extpl)
            L2Dif_array.append(L2Dif)
            vepc_result_array.append(u_extpl_result)
            theory_array.append(theory)
            it_array.append(it)
            print("num grid " + str(num_grid_dr) + ": L2 extpl error " ,L2_rel_dif_extpl)
        
        
        plot_grid_L2dif = True
        if(plot_grid_L2dif):
            # plt.close("all")
            import matplotlib
            from matplotlib import ticker
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True) 
            formatter.set_powerlimits((-1,1)) 
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(num_grid_dr_array,  L2Dif_array_extpl, label = "after extrapolation")
            ax.plot(num_grid_dr_array,  L2Dif_array_extpl,"b.")
            ax.set_yscale("log",basey = 10)
            ax.set_xscale("log",basex = 2)
            # plt.plot(np.log(grid_array),  np.log(L2Dif_array_extpl), label = "after extrapolation")
            # plt.plot(np.log(grid_array),  np.log(L2Dif_array_extpl),"b.")
            plt.plot(num_grid_dr_array,  L2Dif_array, label = "before extrapolation")
            plt.plot(num_grid_dr_array,  L2Dif_array,"r.")
            ax.set_yticks([1e-3,5e-4,1e-4])
            ax.set_xticks([0.1,1,10])
            ax.set_xlabel(r"\delta \phi (*dh)")
            ax.set_ylabel("L2 relative error")
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            from decimal import Decimal
            ax.set_title("test case" + test_case +"{:2}".format(N_grid))
            plt.legend()
        
        
        import h5py as h5
        store_file = True
        if(store_file):
            filename = "\\2d_singular\\sigma_test\\test_case_"+test_case+"_rlx_N"+"{:2}".format(N_grid)+ "_noQ.h5"
            if(cwd[-11:] == '2d_singular'):
                filename = cwd + filename[12:]
            else:
                filename = cwd + filename
            print(filename)

                
            store_data_file = h5.File(filename,"w")
            store_data_file.create_dataset("Ngrid",data = grid_array)
            store_data_file.create_dataset("L2",data = L2Dif_array)
            store_data_file.create_dataset("L2extpl",data = L2Dif_array_extpl)
            store_data_file.create_dataset("sigma_r0_param_eta",data = np.array([sigma,r0,param,eta]))
            for i in range(len(vepc_result_array)):
                store_data_file.create_dataset("vepc" + str(i),data = vepc_result_array[i])
                store_data_file.create_dataset("theory" + str(i),data = theory_array[i])
            store_data_file.create_dataset("it",data = it_array)
            store_data_file.create_dataset("sigma",data = sigma_array)
            store_data_file.create_dataset("num_grid_dr",data = num_grid_dr_array)
            
            store_data_file.close()
            
additional_test = False
if(additional_test):
    test_ver = "2-3"
    N_grid = 128
    u_init, (xmesh,ymesh), h = jns.setup_grid(N_grid)
    r = mhf.XYtoR(xmesh,ymesh)
    r0 = np.sqrt(np.e / 4)
    
    def get_phi(rho_):
        frame_full = mhf.get_frame_n(rho_)
        frame_nobnd = jns.no_boundary(rho_)
        d_rho = np.max(rho_ * (frame_full - frame_nobnd))
        print(d_rho / np.max(rho_))
        
        return rho - d_rho
    
    if(test_ver == "2-1"):
        rho = -(r**2 - r0**2)
        rhoc = r0
        coef = np.copy(rho)
        phi = get_phi(rho)
        param = 1.0
        theory = ymesh + param * np.sin(xmesh)
        boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
        rhs = -(2*(ymesh + param * xmesh* np.cos(xmesh)) - param*(r**2 - r0**2)*np.sin(xmesh))
    
    if(test_ver == "2-2"):
        rho = -(r - r0)
        rhoc = r0
        coef = np.copy(rho)
        phi = get_phi(rho)
        param = 1.0
        theory = ymesh + param * np.sin(xmesh)
        boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
        rhs = -((ymesh + param * xmesh* np.cos(xmesh)) - param*r*(r - r0)*np.sin(xmesh))/ (r + mhf.singular_null)
              
    if(test_ver == "2-3"):
        rho = -(r**0.5 - r0**0.5)
        rhoc = r0
        coef = np.copy(rho)
        phi = get_phi(rho)
        param = 1.0
        theory = ymesh + param * np.sin(xmesh)
        boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
        rhs = -(ymesh + param * xmesh* np.cos(xmesh))/ (2*r**1.5 + mhf.singular_null) - param*rho*np.sin(xmesh)
                 
    
    if(test_ver == "3-1"):
        rho = -(r**2 - r0**2)
        rhoc = r0**2
        coef = np.copy(rho)
        ecc = 1.2
        rp = mhf.XYtoR(ecc*xmesh,ymesh)
        rho = -(rp**2 - r0**2)
        rhoc = np.max(np.abs(rho))
        phi = get_phi(rho)
        coef = rho
        theory = (1 + np.sin(xmesh)*np.cos(ymesh))
        boundary = (ecc**2*xmesh*np.cos(xmesh)*np.cos(ymesh) - ymesh*np.sin(xmesh)*np.sin(ymesh)) \
                        / (mhf.XYtoR(ecc**2*xmesh,ymesh) + mhf.singular_null)
        rhs = -2*ecc**2*xmesh*np.cos(xmesh)*np.cos(ymesh) + \
            2*np.sin(xmesh)*((rp**2-r0**2)*np.cos(ymesh)+ymesh*np.sin(ymesh))
           
        
    
    if(test_ver == "3-2"):
        rho = -(r - r0)
        rhoc = r0
        coef = rho*np.exp(r)
        phi = get_phi(rho)
        param = 1.0
        theory = ymesh + param * np.sin(xmesh)
        boundary = (ymesh + param * xmesh * np.cos(xmesh)) / (r + mhf.singular_null)
        rhs_1 = (ymesh + param * xmesh* np.cos(xmesh))*(1-r0+r) / (r + mhf.singular_null)
        rhs_2 = -param * (r - r0) * np.sin(xmesh)
        rhs = -np.exp(r)*(rhs_1 + rhs_2)
    frame = mhf.get_frame_n(phi)
    frame_full =mhf.get_frame_n(rho)
    ux, uy = mhf.grad(theory,h,h)
    rhs_theory = mhf.div(coef*ux,coef*uy,h,h)
    test = [True, True]
    if(test[0]):
            mhf3d.plot2d_compare(rhs,rhs_theory,frame_full,"rhs")
            
    # test boundary
    if(test[1]):
        mhf3d.plot2d_compare(boundary,mhf.grad_n_n(theory,phi,h,h),jns.get_N1(phi),"boundary")
                    
    maxIter = 200
    it_multi = 10
    eta = 1.0e-4
    rlx = 0.1
    start_run = False
    if(start_run):
        u_extpl_result, it = jns.stm_coef_Neumann(u_init, (xmesh,ymesh),phi,rho,rhs,coef,\
                                            theory*frame, boundary, maxIter,eta,rlx)
        maxDif_extpl, L2Dif_extpl = mhf.get_error_N(u_extpl_result, theory, frame_full,(False,False))
