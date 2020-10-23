# -*- coding: utf-8 -*-
"""
This file contains a realistic 3D binary neutron stars test case

Created on Thu May 7 10:22:12 2020

@author: Bing-Jyun Tsao (btsao2@illinois.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os, sys
import time
import math
cwd = os.getcwd()

sys.path.append(cwd + "//..//")
import mesh_helper_functions_3D as mhf3d
import source_term_method_3D as stm3d


if(__name__ == "__main__"):
    plt.close("all")
     
    start_time = time.time()
    
    
    def get_omega_cross_r(r,omega):
        xmesh, ymesh = r
        return -omega * ymesh, omega * xmesh
    
    
    def get_theory(sol,phi):
        isIn = mhf3d.get_frame(phi)
        sol = isIn*(sol)
        sol_zero = sol - np.sum(sol) / np.sum(isIn) *isIn
        return sol_zero
    
    def no_boundary(phi):
        isIn = mhf3d.get_frame(phi)
        N1 = stm3d.get_N1(phi)
        N2 = stm3d.get_N2(phi)
        return isIn * (1-N2)
    
    def get_phi(rho_):
        frame_full = mhf3d.get_frame(rho_)
        frame_nobnd = no_boundary(rho_)
        d_rho = np.max(rho_ * (frame_full - frame_nobnd))
        print(d_rho / np.max(rho_))
        
        return rho - d_rho
    
    plt.close("all")
    bns_data_3d = h5.File(cwd+"\\bns_3d_data\\bns_data_3d.h5","r")
    
    # this is the omega (without hat)
    omega_0 = 9.075651185344492E-03 # 1.842121059006711E+03 [rad/sec]
    Distance = 3.046718301084026E+01 # 45 km
    R0 = 1.218687320433609E+01 # length normalization constant
    
    # taking care of normalization
    omega_hat = omega_0 *R0
    
    
    x_data = np.array(bns_data_3d.get("x"),dtype = float)
    y_data = np.array(bns_data_3d.get("y"),dtype = float)
    dx_data = x_data[0,1,0] - x_data[0,0,0]
    dy_data = y_data[1,0,0] - y_data[0,0,0]
    h_data = dx_data
    N_data = len(x_data)
    z_data = np.zeros_like(x_data)
    for i in range(N_data):
        z_data[:,:,i] = np.ones_like(x_data[:,:,i]) * (i - N_data/2) * dx_data
    dz_data = z_data[0,0,1] - z_data[0,0,0]
    
    xmin, xmax = np.min(x_data), np.max(x_data)
    ymin, ymax = np.min(y_data), np.max(y_data)
    zmin, zmax = np.min(z_data), np.max(z_data)
    

    N_grid = 60
    N_x, N_y, N_z = N_grid, N_grid, N_grid
    x_line = np.linspace(xmin, xmax, N_x,endpoint =True)
    y_line = np.linspace(ymin, ymax, N_y,endpoint =True)
    z_line = np.linspace(zmin, zmax, N_z,endpoint =True)
    
    
    xmesh, ymesh, zmesh = np.meshgrid(x_line, y_line, z_line)
    dx = xmesh[0,1,0] - xmesh[0,0,0]
    dy = ymesh[1,0,0] - ymesh[0,0,0]
    dz = zmesh[0,0,1] - zmesh[0,0,0]
    h = dx
    mesh0 = (x_data, y_data, z_data)
    mesh = (xmesh, ymesh, zmesh)
    
    
    vepc_data = np.array(bns_data_3d.get("vepc"),dtype = float)
    vepc = stm3d.interpolation_N(mesh0, mesh, vepc_data)
    
    rho_data = np.array(bns_data_3d.get("rhoc"),dtype = float)
    rho = stm3d.interpolation_N(mesh0, mesh, rho_data)
    
    
    vxco_data = np.array(bns_data_3d.get("vxco"),dtype = float)
    vyco_data = np.array(bns_data_3d.get("vyco"),dtype = float)
    vzco_data = np.array(bns_data_3d.get("vzco"),dtype = float)
    vxco = stm3d.interpolation_N(mesh0, mesh, vxco_data)
    vyco = stm3d.interpolation_N(mesh0, mesh, vyco_data)
    vzco = stm3d.interpolation_N(mesh0, mesh, vzco_data)
    
    
    
    vxi_data = np.array(bns_data_3d.get("vxi"),dtype = float)
    vyi_data = np.array(bns_data_3d.get("vyi"),dtype = float)
    vzi_data = np.array(bns_data_3d.get("vzi"),dtype = float)
    vxi = stm3d.interpolation_N(mesh0, mesh, vxi_data)
    vyi = stm3d.interpolation_N(mesh0, mesh, vyi_data)
    vzi = stm3d.interpolation_N(mesh0, mesh, vzi_data)
    
    vxc_data = np.array(bns_data_3d.get("vxc"),dtype = float)
    vyc_data = np.array(bns_data_3d.get("vyc"),dtype = float)
    vzc_data = np.array(bns_data_3d.get("vzc"),dtype = float)
    vxc = stm3d.interpolation_N(mesh0, mesh, vxc_data)
    vyc = stm3d.interpolation_N(mesh0, mesh, vyc_data)
    vzc = stm3d.interpolation_N(mesh0, mesh, vzc_data)
    
    #actually not bvxd but bvxc
    bvxd_data = np.array(bns_data_3d.get("bvxc"),dtype = float)
    bvyd_data = np.array(bns_data_3d.get("bvyc"),dtype = float)
    bvzd_data = np.array(bns_data_3d.get("bvzc"),dtype = float)
    bvxd = stm3d.interpolation_N(mesh0, mesh, bvxd_data)
    bvyd = stm3d.interpolation_N(mesh0, mesh, bvyd_data)
    bvzd = stm3d.interpolation_N(mesh0, mesh, bvzd_data)
    
    ut_data = np.array(bns_data_3d.get("utc"),dtype = float)
    hhc_data = np.array(bns_data_3d.get("hhc"),dtype = float)
    alpha_data = np.array(bns_data_3d.get("alphc"),dtype = float)
    
    psi_data = np.array(bns_data_3d.get("psic"),dtype = float)
    
    ut = stm3d.interpolation_N(mesh0, mesh, ut_data)
    hhc = stm3d.interpolation_N(mesh0, mesh, hhc_data)
    alpha = stm3d.interpolation_N(mesh0, mesh, alpha_data)
    psi = stm3d.interpolation_N(mesh0, mesh,psi_data)
      
    
    central_density = np.max(np.abs(rho))
    rho_cut_off = 1E-11 * central_density
    rho = (rho - rho_cut_off)
    isIn_full = mhf3d.get_frame(-rho)
    wx, wy = get_omega_cross_r((xmesh,ymesh),omega_hat)
    rhox, rhoy, rhoz = mhf3d.grad(rho, dx, dy, dz)
    f = rhox*wx + rhoy * wy
    rho_norm = mhf3d.abs_grad(rho, dx,dy,dz)
    rhs = f
    ux, uy, uz = mhf3d.grad(vepc,dx,dy,dz)
    rux, ruy, ruz = ux*rho, uy*rho, uz*rho
    
    omegax = bvxd + wx
    omegay = bvyd + wy
    omegaz = bvzd
    
    
    test_3d = True
    if(test_3d):
        # maximum iteration number for the source term method
        maxIter = 500
        # the total iteration number N for Jacobi solver = it_multi * N_grid**2
        it_multi = 10
        
        grid_size = len(xmesh)
        half_grid_size = int(0.5 * grid_size)
        h = dx
        u_init = np.zeros_like(xmesh)
        
        # 2. set up required parameters using functions from the test case
        arh = alpha*rho / (hhc+mhf3d.singular_null)
        dx_psi, dy_psi,dz_psi = mhf3d.grad(psi,dx,dy,dz)
        dx_arh, dy_arh, dz_arh = mhf3d.grad(arh,dx,dy,dz)
        dx_hut, dy_hut, dz_hut = mhf3d.grad(hhc*ut,dx,dy,dz)
        dx_rho, dy_rho, dz_rho = mhf3d.grad(rho,dx,dy,dz)
        cx = -2 / (psi + mhf3d.singular_null) * dx_psi - dx_arh / (arh + mhf3d.singular_null)
        cy = -2 / (psi + mhf3d.singular_null) * dy_psi - dy_arh / (arh + mhf3d.singular_null)
        cz = -2 / (psi + mhf3d.singular_null) * dz_psi - dz_arh / (arh + mhf3d.singular_null)
        
        rhs_term1 = psi**4 * (omegax*dx_arh + omegay*dy_arh + omegaz*dz_arh) * hhc * ut / (arh + mhf3d.singular_null)
        rhs_term2 = psi**4 * (omegax*dx_hut + omegay*dy_hut + omegaz*dz_hut) 
        
        phi_test = get_phi(rho)
        sol_test = vepc*1.0
        rho_test = rho
        rhs_test = psi**6 * mhf3d.grad_dot(ut*alpha*rho, (omegax,omegay,omegaz),dx,dy,dz)
        coef_test = (psi**2 * alpha * rho)/(hhc + mhf3d.singular_null)
        isIn_small = mhf3d.get_frame(phi_test)
        isIn_large = mhf3d.get_frame(rho_test)
            
        grad_rho_dot_Phi = psi**4 * (omegax*dx_rho + omegay*dy_rho + omegaz*dz_rho)  * hhc * ut
        boundary2 = - grad_rho_dot_Phi/(np.sqrt(mhf3d.grad_dot_grad(rho_test,rho_test,h,h,h))+mhf3d.singular_null)
        
        # 3. run the coefficient poisson solver
        u_result_3d = stm3d.stm_coef_Neumann_3d(u_init, it_multi, (xmesh,ymesh,zmesh), \
                            phi_test, rho_test, rhs_test, coef_test, sol_test, boundary2, maxIter)

        
        # 4. calculate the error with theoretical value
        theory_test = get_theory(sol_test,-rho_test)
        maxDif,L2Dif = mhf3d.get_error_N(u_result_3d, theory_test , mhf3d.get_frame(rho_test))
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("The code takes %f sec" % time_elapsed)
        
        layer = half_grid_size
        N2 = stm3d.get_N2(rho)
        N3 = stm3d.get_N3(rho)
        
        
        write_result = True
        if(write_result):
            result_file = h5.File(cwd+"\\result\\bns3d_result_final_noQ.h5","w")
            # result_file = h5.File(cwd+"\\3d_bns\\bns3d_result_dh" + name[idx] + ".h5","w")
            result_file.create_dataset("vepc",data = u_result_3d)
            result_file.create_dataset("theory", data = theory_test)
            result_file.create_dataset("vxtheory", data = vxc*psi**4)
            result_file.create_dataset("vytheory", data = vyc*psi**4)
            result_file.create_dataset("vztheory", data = vzc*psi**4)
            result_file.create_dataset("phi",data = phi_test)
            result_file.create_dataset("rho",data = rho_test)
            result_file.create_dataset("x", data = xmesh)
            result_file.create_dataset("y", data = ymesh)
            result_file.create_dataset("z", data = zmesh)
            result_file.close()
            print("file saved")
        
            
        