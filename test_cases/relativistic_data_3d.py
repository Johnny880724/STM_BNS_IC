# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:22:12 2020

@author: Johnny Tsao
"""

import mesh_helper_functions_3d as mhf3d
import mesh_helper_functions as mhf
import test_cases_3d as tc
import jacobi_newtonian_solver_3d as jns1
import jacobi_newtonian_solver_3d_n as jns2
import jacobi_newtonian_solver as jns
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import time
import math
cwd = os.getcwd()

if(__name__ == "__main__"):
    plt.close("all")
     
    start_time = time.time()
    
    
    def get_omega_cross_r(r,omega):
        xmesh, ymesh = r
        return -omega * ymesh, omega * xmesh
    
    
    def get_theory(sol,phi):
        isIn = mhf.get_frame(phi)
        sol = isIn*(sol)
        sol_zero = sol - np.sum(sol) / np.sum(isIn) *isIn
        return sol_zero
    
    def no_boundary(phi):
        isIn = mhf3d.get_frame_n(phi)
        N1 = jns.get_N1(phi)
        N2 = jns.get_N2(phi)
        return isIn * (1-N2)
    
    def get_phi(rho_):
        frame_full = mhf.get_frame_n(rho_)
        frame_nobnd = no_boundary(rho_)
        d_rho = np.max(rho_ * (frame_full - frame_nobnd))
        print(d_rho / np.max(rho_))
        
        return rho - d_rho
    
    plt.close("all")
    bns_data_3d = h5.File(cwd+"\\bns_data\\bns_data_3d.h5","r")
    
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
    vepc = jns2.interpolation_N(mesh0, mesh, vepc_data)
    
    rho_data = np.array(bns_data_3d.get("rhoc"),dtype = float)
    rho = jns2.interpolation_N(mesh0, mesh, rho_data)
    
    
    vxco_data = np.array(bns_data_3d.get("vxco"),dtype = float)
    vyco_data = np.array(bns_data_3d.get("vyco"),dtype = float)
    vzco_data = np.array(bns_data_3d.get("vzco"),dtype = float)
    vxco = jns2.interpolation_N(mesh0, mesh, vxco_data)
    vyco = jns2.interpolation_N(mesh0, mesh, vyco_data)
    vzco = jns2.interpolation_N(mesh0, mesh, vzco_data)
    
    
    
    vxi_data = np.array(bns_data_3d.get("vxi"),dtype = float)
    vyi_data = np.array(bns_data_3d.get("vyi"),dtype = float)
    vzi_data = np.array(bns_data_3d.get("vzi"),dtype = float)
    vxi = jns2.interpolation_N(mesh0, mesh, vxi_data)
    vyi = jns2.interpolation_N(mesh0, mesh, vyi_data)
    vzi = jns2.interpolation_N(mesh0, mesh, vzi_data)
    
    vxc_data = np.array(bns_data_3d.get("vxc"),dtype = float)
    vyc_data = np.array(bns_data_3d.get("vyc"),dtype = float)
    vzc_data = np.array(bns_data_3d.get("vzc"),dtype = float)
    vxc = jns2.interpolation_N(mesh0, mesh, vxc_data)
    vyc = jns2.interpolation_N(mesh0, mesh, vyc_data)
    vzc = jns2.interpolation_N(mesh0, mesh, vzc_data)
    
    #actually not bvxd but bvxc
    bvxd_data = np.array(bns_data_3d.get("bvxc"),dtype = float)
    bvyd_data = np.array(bns_data_3d.get("bvyc"),dtype = float)
    bvzd_data = np.array(bns_data_3d.get("bvzc"),dtype = float)
    bvxd = jns2.interpolation_N(mesh0, mesh, bvxd_data)
    bvyd = jns2.interpolation_N(mesh0, mesh, bvyd_data)
    bvzd = jns2.interpolation_N(mesh0, mesh, bvzd_data)
    
    ut_data = np.array(bns_data_3d.get("utc"),dtype = float)
    hhc_data = np.array(bns_data_3d.get("hhc"),dtype = float)
    alpha_data = np.array(bns_data_3d.get("alphc"),dtype = float)
    
    psi_data = np.array(bns_data_3d.get("psic"),dtype = float)
    
    ut = jns2.interpolation_N(mesh0, mesh, ut_data)
    hhc = jns2.interpolation_N(mesh0, mesh, hhc_data)
    alpha = jns2.interpolation_N(mesh0, mesh, alpha_data)
    psi = jns2.interpolation_N(mesh0, mesh,psi_data)
      
    
#    plt.matshow(vepc0)
#    plt.matshow(rho0)
#    plt.matshow(x0)
#    plt.matshow(y0)
    test2d = True
    test3d = True
    
    central_density = np.max(np.abs(rho))
    rho_cut_off = 1E-11 * central_density
    dh = central_density * 0.1
    rho = (rho - rho_cut_off)
    phi = rho - dh
    isIn_full = mhf3d.get_frame(-rho)
    isIn_0 = mhf3d.get_frame(-phi)
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
    
    vx_plot = vxi*isIn_0
    vy_plot = vyi*isIn_0
    vz_plot = vzi*isIn_0
    
    
    show_frame = no_boundary(phi)
    
    test_layer = int(0.5*N_grid)
    
    ## test1 u = omega cross r  + v
    # u = inertial (vxi, vyi, vzi)
    # v = rotating (vx0, vy0, vz0)
    # this is correct
    test1= False
    if(test1):
        test1x = (vxi - vxco)
        test1y = (vyi - vyco)
        test1z = (vzi - vzco)
        mhf3d.plot2d_compare(test1x[:,:,test_layer], wx[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test1y[:,:,test_layer], wy[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test1z[:,:,test_layer], np.zeros_like(test1z[:,:,test_layer]))
    
    
    ## test grad(phi) = u (vxi, vyi, vzi)
    test2=False
    if(test2):
        test2x,test2y,test2z = (mhf3d.grad(vepc,dx,dy,dz))
        mhf3d.plot2d_compare(test2x[:,:,test_layer], (vxi*psi**4/R0**2)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test2y[:,:,test_layer], vyi[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test2z[:,:,test_layer], vzi[:,:,test_layer],show_frame[:,:,test_layer])
    
    ##test rhs = lhs
    test3 = False
    if(test3):
        scale_up = 1e7
        mult = 0.68e5
        rhs_plot = (-isIn0*rhs*scale_up)
        lhs_plot = (isIn0*mhf3d.div(rux, ruy, ruz,dx,dy,dz))
        lhs_plot2 = (isIn0*mhf3d.div(rho0*vxi, rho0*vyi, rho0*vzi,dx,dy,dz))*mult
        mhf3d.plot2d_compare(rhs_plot[:,:,test_layer], lhs_plot2[:,:,test_layer])
    #    mhf3d.plot2d(rhs_plot[:,:,test_layer] / (lhs_plot[:,:,test_layer]+mhf3d.singular_null))
    
    ## test vc = grad(phi) / psi**4
    ## now this is correct
    test4 = False
    if(test4):
        test4x,test4y,test4z = (mhf3d.grad(vepc0,dx,dy,dz))
        mhf3d.plot2d_compare(test4x[:,:,test_layer], (vxc*psi**4)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test4y[:,:,test_layer], (vyc*psi**4)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test4z[:,:,test_layer], (vzc*psi**4)[:,:,test_layer],show_frame[:,:,test_layer])
    
    ## test vi = vc/(h ut psi**4) - beta
    ## now this is correct
    test5 = False
    if(test5):
        test5x = vxc /(hhc*ut + mhf3d.singular_null) - bvxd
        test5y = vyc /(hhc*ut + mhf3d.singular_null) - bvyd
        test5z = vzc /(hhc*ut + mhf3d.singular_null) - bvzd
        mhf3d.plot2d_compare(test5x[:,:,test_layer], (vxi)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test5y[:,:,test_layer], (vyi)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test5z[:,:,test_layer], (vzi)[:,:,test_layer],show_frame[:,:,test_layer])
        
    test6 = False
    if(test6):
        test6x = vxi + bvxd
        test6y = vyi + bvyd
        test6z = vzi + bvzd
        mhf3d.plot2d_compare(test6x[:,:,test_layer], (vx0)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test6y[:,:,test_layer], (vy0)[:,:,test_layer],show_frame[:,:,test_layer])
        mhf3d.plot2d_compare(test6z[:,:,test_layer], (vz0)[:,:,test_layer],show_frame[:,:,test_layer])
        
    ## equation 57
    # D^2 Phi = -2/psi d_i psi d^i Phi + psi^4 
    test7 = False
    if(test7):
        change = R0
        term1 = -2 / (psi + mhf3d.singular_null) * mhf3d.grad_dot_grad(psi,vepc,dx,dy,dz)
        term2 = psi**4 * mhf3d.grad_dot(hhc*ut,(omegax,omegay,omegaz),dx,dy,dz)
        
        #do not take derivative with log
#        log_arh = mhf3d.log_frame(alpha*rho0/(h0+mhf3d.singular_null), isIn_full)
#        lrx, lry, lrz = mhf3d.grad(log_arh,dx,dy,dz)
        
        arh = alpha*rho/(hhc+mhf3d.singular_null)
        lrx, lry, lrz = mhf3d.grad(arh,dx,dy,dz) / (arh+mhf3d.singular_null)
        
#        mhf3d.plot2d_compare(lrx[:,:,test_layer], lrx2[:,:,test_layer],show_frame[:,:,test_layer])
        
        term3 = psi**4 * hhc* ut * (omegax*lrx + omegay*lry + omegaz*lrz)
        term4 = - mhf3d.grad_dot(vepc,(lrx,lry,lrz),dx,dy,dz)
#        mhf3d.plot2d(term3[:,:,test_layer],"",show_frame[:,:,test_layer])
        test7_rhs = term1 + term2 + term3 + term4
#        test7_rhs = 0*term1 + 0*term2 + 0*term3 + term4
        test7_lhs = mhf3d.laplace(vepc,dx,dy,dz)
        mhf3d.plot2d_compare(test7_rhs[:,:,test_layer], test7_lhs[:,:,test_layer],show_frame[:,:,test_layer])
        #mhf3d.plot2d_compare((test7_rhs -term1 - term4)[:,:,test_layer], (test7_lhs - term1 - term4)[:,:,test_layer],isIn0[:,:,test_layer])
    
    test8 = False
    if(test8):
        test8_lhs = mhf3d.grad_dot_grad(vepc0, rho0, dx, dy, dz)
        test8_rhs = psi**4 * h0 * ut * mhf3d.grad_dot(rho0,(omegax,omegay,omegaz),dx,dy,dz)
#        mhf3d.plot2d_compare(test8_rhs[:,:,test_layer], test8_lhs[:,:,test_layer],(isIn0)[:,:,test_layer])
        mhf3d.plot2d_compare((test8_rhs)[:,:,test_layer], test8_lhs[:,:,test_layer],(show_frame)[:,:,test_layer])
        
        
    test9 = False
    if(test9):
        arh = alpha*rho0/(h0+mhf3d.singular_null)
        dx_psi, dy_psi,dz_psi = mhf3d.grad(psi,dx,dy,dz)
        dx_arh, dy_arh, dz_arh = mhf3d.grad(arh,dx,dy,dz)
        dx_hut, dy_hut, dz_hut = mhf3d.grad(h0*ut,dx,dy,dz)
        dx_rho, dy_rho, dz_rho = mhf3d.grad(rho0,dx,dy,dz)
        cx = -2 / (psi + mhf3d.singular_null) * dx_psi - dx_arh / (arh + mhf3d.singular_null)
        cy = -2 / (psi + mhf3d.singular_null) * dy_psi - dy_arh / (arh + mhf3d.singular_null)
        cz = -2 / (psi + mhf3d.singular_null) * dz_psi - dz_arh / (arh + mhf3d.singular_null)
        
        rhs_term1 = psi**4 * (omegax*dx_arh + omegay*dy_arh + omegaz*dz_arh) * h0 * ut / (arh + mhf3d.singular_null)
        rhs_term2 = psi**4 * (omegax*dx_hut + omegay*dy_hut + omegaz*dz_hut) 
        phi2 = rho0 - dh
        sol2 = vepc0*1.0
        rho2 = rho0
        rhs2 = rhs_term1 + rhs_term2
        test9_rhs = rhs2
        
        test9_lhs = mhf3d.laplace(vepc0,dx,dy,dz) - mhf3d.grad_dot(vepc0,(cx,cy,cz),dx,dy,dz)
        mhf3d.plot2d_compare(test9_rhs[:,:,test_layer], test9_lhs[:,:,test_layer],show_frame[:,:,test_layer])
          
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
        u_cur1 = np.copy(u_init)
        u_cur2 = np.copy(u_init)
#        u_init[:,:,0] = vepc0[:,:,0]
#        u_init[:,:,2] = vepc0[:,:,2]
        
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
        
#        name = ["40","30","25","20","18","16","15","14","12","10","08","06","05"]
        # name = ["07","09","11","13"]
        name = ["none"]
        for idx in range(len(name)):
            # print("dh = 0." + name[idx] + " rho 0")
#            phi2 = rho - 0.5*dh
            # phi2 = rho - int(name[idx])*0.1**len(name[idx])*dh*10
            phi2 = get_phi(rho)
            sol2 = vepc*1.0
            rho2 = rho
            rhs2 = psi**6 * mhf3d.grad_dot(ut*alpha*rho, (omegax,omegay,omegaz),dx,dy,dz)
            coef2 = (psi**2 * alpha * rho)/(hhc + mhf3d.singular_null)
            isIn2 = mhf3d.get_frame_n(phi2)
            isIn_small = mhf3d.get_frame_n(phi2)
            isIn_large = mhf3d.get_frame_n(rho2)
                
            #div(coef*grad(Phi)) = rhs
            test_simulation1 = False
            if(test_simulation1):
                dx_Phi, dy_Phi, dz_Phi = mhf3d.grad(vepc,dx,dy,dz)
                lhs = mhf3d.div(coef2*dx_Phi, coef2*dy_Phi, coef2*dz_Phi, dx,dy,dz)
                rhs = rhs2*1.0
                #mhf3d.plot2d_compare(lhs[:,:,test_layer],rhs[:,:,test_layer],show_frame[:,:,test_layer])
                mhf3d.plot2d_compare(lhs[:,test_layer,:],rhs[:,test_layer,:],isIn_small[:,test_layer,:]) 
                mhf3d.plot2d_compare(lhs[test_layer,:,:],rhs[test_layer,:,:],isIn_small[test_layer,:,:]) 
            # laplace(Phi) = (rhs - grad(coef) dot grad(Phi)) / coef
            test_simulation2 = False
            if(test_simulation2):
                lhs = mhf3d.laplace(vepc,dx,dy,dz)
                rhs = (rhs2 - mhf3d.grad_dot_grad(coef2,vepc,dx,dy,dz))/(coef2 + mhf3d.singular_null)
                
                mhf3d.plot2d_compare(lhs[:,:,test_layer],(rhs)[:,:,test_layer],isIn_small[:,:,test_layer])
            
            test_product_rule = False
            if(test_product_rule):
                dhu = mhf3d.grad_dot(hhc*ut,(omegax,omegay,omegaz),dx,dy,dz)
                darh = mhf3d.grad_dot(alpha*rho/(hhc+mhf3d.singular_null),(omegax,omegay,omegaz),dx,dy,dz)
                daru = mhf3d.grad_dot(ut*alpha*rho, (omegax,omegay,omegaz),dx,dy,dz)
                term2_pd = psi**4*coef2*(dhu)
                term3_pd = psi**4*coef2*(darh * hhc*ut / (arh+mhf3d.singular_null))
                lhs_pd = term2_pd + term3_pd
                rhs_pd = psi**6*daru
                #mhf3d.plot2d_compare(lhs_pd[:,:,test_layer],rhs_pd[:,:,test_layer],isIn_small[:,:,test_layer])
                mhf3d.plot2d_compare(term3_pd[:,:,test_layer],(term3*coef2)[:,:,test_layer],isIn_small[:,:,test_layer])
               
            
            test_laplacian = False
            if(test_laplacian):
                lhs = mhf3d.laplace(vepc0,dx,dy,dz) * rho0 + mhf3d.grad_dot_grad(coef2,vepc0,dx,dy,dz)
                dx_Phi, dy_Phi, dz_Phi = mhf3d.grad(vepc0,dx,dy,dz)
                rhs = mhf3d.div(coef2*dx_Phi, coef2*dy_Phi, coef2*dz_Phi, dx,dy,dz)
                mhf3d.plot2d_compare(lhs[:,:,test_layer],rhs[:,:,test_layer],show_frame[:,:,test_layer])
            
            grad_rho_dot_Phi = psi**4 * (omegax*dx_rho + omegay*dy_rho + omegaz*dz_rho)  * hhc * ut
            boundary2 = - grad_rho_dot_Phi/(np.sqrt(mhf3d.grad_dot_grad(rho2,rho2,h,h,h))+mhf3d.singular_null)
            
            test_boundary = False
            if(test_boundary):
                lhs = mhf3d.grad_n_n(sol2,phi2,dx,dy,dz)
                rhs = boundary2 * 1.0
                boundary_frame = jns.get_N2(phi2)
                mhf3d.plot2d_compare(lhs[:,:,test_layer],rhs[:,:,test_layer],boundary_frame[:,:,test_layer])
            
            # 3. run the coefficient poisson solver
            u_extpl_result2 = jns2.coef_poisson_jacobi_source_term_Neumann_relativistic(u_init, it_multi, (xmesh,ymesh,zmesh), phi2, rho2, rhs2, coef2, sol2, boundary2, maxIter)
            u_cur2 = np.copy(u_extpl_result2)
            
            # 4. calculate the error with theoretical value
            theory2 = get_theory(sol2,-rho2)
            maxDif,L2Dif = mhf3d.get_error_N(u_extpl_result2, theory2 , mhf3d.get_frame_n(rho2))
            
            end_time = time.time()
            time_elapsed = end_time - start_time
            print("The code takes %f sec" % time_elapsed)
            
            # phi_smaller = phi2 - 0.5*dh
            # isIn_smaller = mhf3d.get_frame_n(phi_smaller)
            isIn_large_nobound = no_boundary(rho2)
            layer = half_grid_size
            N2 = jns2.get_N2(rho)
            N3 = jns.get_N3(rho)
    #        plt.matshow(N2[:,:,test_layer])
    #        mhf3d.plot2d_compare(u_extpl_result2[:,:,layer], theory2[:,:,layer],isIn_smaller[:,:,layer])
    #        mhf3d.plot2d_compare(u_extpl_result2[:,:,layer], theory2[:,:,layer],isIn_small[:,:,layer])
            
            plot_3d = True
            if(plot_3d):
                plot_frame = isIn_large*1.0
                for i in range(2):
                    plot_frame = jns2.remove_boundary(plot_frame)
                mhf3d.plot2d_compare(u_extpl_result2[:,:,layer], theory2[:,:,layer],plot_frame[:,:,layer],"velocity potential (x-y plane)",["x","y"])
                mhf3d.plot2d_compare(u_extpl_result2[:,layer,:], theory2[:,layer,:],plot_frame[:,layer,:],"velocity potential (y-z plane)",["z","y"])
                mhf3d.plot2d_compare(u_extpl_result2[layer,:,:], theory2[layer,:,:],plot_frame[layer,:,:],"velocity potential (z-x plane)",["z","x"])
            
            plot_v = False
            if(plot_v):
                vxi_result, vyi_result, vzi_result = mhf3d.grad(u_extpl_result2,dx,dy,dz)
                
                fig = plt.figure(figsize = (8,6))
                ax = fig.gca()
                ax.plot(vyi_result[layer,:,layer],label = "result ($\partial_y \Phi$)")
                ax.plot(vyc[layer,:,layer]*psi[layer,:,layer]**4, label = "theory ($v_y$)")
                ax.legend()
                ax.set_xlabel("x")
                ax.set_ylabel("$v_y$")
                ax.set_title("$v_y$  - x plot")
                
                fig = plt.figure(figsize = (8,6))
                ax = fig.gca()
                ax.plot(vyi_result[layer,layer,:],label = "result ($\partial_y \Phi$)")
                ax.plot(vyc[layer,layer,:]*psi[layer,layer,:]**4, label = "theory ($v_y$)")
                ax.legend()
                ax.set_xlabel("z")
                ax.set_ylabel("$v_y$")
                ax.set_title("$v_y$  - z plot")
                
                fig = plt.figure(figsize = (8,6))
                ax = fig.gca()
                ax.plot(vxi_result[:,layer,layer],label = "result ($\partial_x \Phi$)")
                ax.plot(vxc[:,layer,layer]*psi[:,layer,layer]**4, label = "theory ($v_x$)")
                ax.legend()
                ax.set_xlabel("y")
                ax.set_ylabel("$v_x$")   
                ax.set_title("$v_x$  - y plot")
                
                fig = plt.figure(figsize = (8,6))
                ax = fig.gca()
                ax.plot(vzi_result[:,layer,layer],label = "result ($\partial_z \Phi$)")
                ax.plot(vzc[:,layer,layer]*psi[:,layer,layer]**4, label = "theory ($v_z$)")
                ax.legend()
                ax.set_xlabel("y")
                ax.set_ylabel("$v_z$")  
                ax.set_title("$v_z$  - y plot")
            
            write_result = True
#             and (not math.isnan(u_extpl_result2[30,30,30]))
            if(write_result):
                result_file = h5.File(cwd+"\\result\\bns3d_result_final_noQ.h5","w")
                # result_file = h5.File(cwd+"\\3d_bns\\bns3d_result_dh" + name[idx] + ".h5","w")
                result_file.create_dataset("vepc",data = u_extpl_result2)
                result_file.create_dataset("theory", data = theory2)
                result_file.create_dataset("vxtheory", data = vxc*psi**4)
                result_file.create_dataset("vytheory", data = vyc*psi**4)
                result_file.create_dataset("vztheory", data = vzc*psi**4)
                result_file.create_dataset("phi",data = phi2)
                result_file.create_dataset("rho",data = rho2)
                result_file.create_dataset("x", data = xmesh)
                result_file.create_dataset("y", data = ymesh)
                result_file.create_dataset("z", data = zmesh)
    #            result_file.create_dataset("time signature", data = np.array([time_elapsed, ])
                result_file.close()
                print("file saved")
        
            
        