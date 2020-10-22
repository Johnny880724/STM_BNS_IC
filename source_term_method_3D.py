        # -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:09:48 2020

@author: Johnny Tsao
"""
import sys
import mesh_helper_functions_3d as mhf3d
import test_cases_3d as tc
import numpy as np
import matplotlib.pyplot as plt

#adding to denominator to avoid 0/0 error
singular_null = mhf3d.singular_null

def setup_grid_3d(N_grid_val = 100, additional = 1):
    # grid dimension
    grid_min = -1.
    grid_max = 1.
    
    global N_grid
    N_grid = N_grid_val
    # grid spacing
    global h
    h = (grid_max - grid_min) / (N_grid) 
    
    # define arrays to hold the x and y coordinates
    xyz = np.linspace(grid_min,grid_max,N_grid + 1)
    global x, y, z
    x,y,z = xyz,xyz,xyz
    global xmesh, ymesh, zmesh
    xmesh_full, ymesh_full, zmesh_full = np.meshgrid(xyz,xyz,xyz)
    mid = int(N_grid / 2)
    xmesh = xmesh_full[:,:, mid - additional : mid + 1 + additional]
    ymesh = ymesh_full[:,:, mid - additional : mid + 1 + additional]
    zmesh = zmesh_full[:,:, mid - additional : mid + 1 + additional]
    # solution
    global u_init
    u_init = np.zeros_like(xmesh)
    return u_init, (xmesh,ymesh,zmesh), h

#discretization of Step, Delta functions
def I(phi):
    return mhf3d.D(phi)*phi

def J(phi):
    return 1./2 * mhf3d.D(phi)*phi**2

def K(phi):
    return 1./6 * mhf3d.D(phi)*phi**3

#Characteristic function of N1
##Chi is 1 if in N1
##       0 if not in N1
def Chi(phi):
    ret = np.zeros_like(phi)
    ret[:-1,:,:] += mhf3d.D(-phi[:-1,:,:]*phi[ 1:,:,:])
    ret[ 1:,:,:] += mhf3d.D(-phi[:-1,:,:]*phi[ 1:,:,:])
    ret[:,:-1,:] += mhf3d.D(-phi[:,:-1,:]*phi[ :,1:,:])
    ret[:, 1:,:] += mhf3d.D(-phi[:,:-1,:]*phi[ :,1:,:])
    ret[:,:,:-1] += mhf3d.D(-phi[:,:,:-1]*phi[ :,:,1:])
    ret[:,:, 1:] += mhf3d.D(-phi[:,:,:-1]*phi[ :,:,1:])
    return np.heaviside(ret,0)

#remove the boundary points
def remove_boundary(domain):
    ret = np.zeros_like(domain)
    ret += domain
    ret[ 1:,  :,  :] *= domain[:-1,  :,  :]
    ret[:-1,  :,  :] *= domain[ 1:,  :,  :]
    ret[  :, 1:,  :] *= domain[  :,:-1,  :]
    ret[  :,:-1,  :] *= domain[  :, 1:,  :]
    ret[  :,  :, 1:] *= domain[  :,  :,:-1]
    ret[  :,  :,:-1] *= domain[  :,  :,1 :]
    return np.heaviside(ret,0)

# return the neighbor and the domain
def get_neighbor(domain):
    ret = np.zeros_like(domain)
    ret += domain
    ret[ 1:,  :,  :] += domain[:-1,  :,  :]
    ret[:-1,  :,  :] += domain[ 1:,  :,  :]
    ret[  :, 1:,  :] += domain[  :,:-1,  :]
    ret[  :,:-1,  :] += domain[  :, 1:,  :]
    ret[  :,  :, 1:] += domain[  :,  :,:-1]
    ret[  :,  :,:-1] += domain[  :,  :,1 :]
    return np.heaviside(ret,0)

# return N1
def get_N1(phi):
    return Chi(phi)

# return N2
def get_N2(phi):
    N1 = get_N1(phi)    
    return get_neighbor(N1)

#return N3
def get_N3(phi):
    N2 = get_N2(phi)
    return get_neighbor(N2)

# return the heaviside function discretized in the paper
def H(phi_,h):
    J_mat = J(phi_)
    K_mat = K(phi_)
    first_term_1 = mhf3d.laplace(J_mat,h,h,h) / (mhf3d.abs_grad(phi_,h,h,h)**2 + singular_null)
    first_term_2 = -(mhf3d.laplace(K_mat,h,h,h) - J_mat*mhf3d.laplace(phi_,h,h,h))*mhf3d.laplace(phi_,h,h,h) / (mhf3d.abs_grad(phi_,h,h,h)**4 + singular_null)
    first_term = first_term_1 + first_term_2
    second_term = mhf3d.D(phi_)
    return Chi(phi_) * first_term + (1-Chi(phi_)) * second_term

# return the delta function discretized in the paper
def delta(phi_,h):
    I_mat = I(phi_)
    J_mat = J(phi_)
    first_term = mhf3d.laplace(I_mat,h,h,h) / (mhf3d.abs_grad(phi_,h,h,h)**2 + singular_null)
    first_term -= (mhf3d.laplace(J_mat,h,h,h) - I_mat*mhf3d.laplace(phi_,h,h,h))*mhf3d.laplace(phi_,h,h,h) / (mhf3d.abs_grad(phi_,h,h,h)**4 + singular_null)
    return Chi(phi_) * first_term
    
# return the source term discretized in the paper
def get_source(a, b, phi_,f_mat_, h_):    
    #Discretization of the source term - formula (7)
    H_h_mat = H(phi_,h_)
    H_mat = mhf3d.D(phi_)
    term1 = mhf3d.laplace(b * H_mat,h_,h_,h_)
    term2 = - H_h_mat * mhf3d.laplace(b, h_, h_,h_)
    term3 = - (a - mhf3d.grad_n_n(b,phi_,h_,h_,h_)) * delta(phi_, h_) * mhf3d.abs_grad(phi_,h_,h_,h_)
    term4 = H_h_mat * f_mat_
    S_mat = term1 + term2 + term3 + term4
    return S_mat

def regularize(mesh_p, mesh):
    reg_min = np.min(mesh)
    reg_max = np.max(mesh)
    too_large = mhf3d.get_frame_n(mesh_p - reg_max)
    too_small = mhf3d.get_frame_n(reg_min - mesh_p)
    mesh_reg = mesh*(1-too_large)*(1-too_small) + too_large * reg_max + too_small * reg_min
    return mesh_reg

#projection algorithm
def projection(mesh_, phi_):
    xmesh, ymesh, zmesh = mesh_
    h = xmesh[0,1,0]-xmesh[0,0,0]
    phi_abs_grad = mhf3d.abs_grad(phi_,h,h,h)
    grad_tup = mhf3d.grad(phi_,h,h,h)
    nx = -grad_tup[0] / (phi_abs_grad + singular_null)
    ny = -grad_tup[1] / (phi_abs_grad + singular_null)
    nz = -grad_tup[2] / (phi_abs_grad + singular_null)
    xp = xmesh + nx * phi_ / (phi_abs_grad + singular_null)
    yp = ymesh + ny * phi_ / (phi_abs_grad + singular_null)
    zp = zmesh + nz * phi_ / (phi_abs_grad + singular_null)
    xp_reg = regularize(xp,xmesh)
    yp_reg = regularize(yp,ymesh)
    zp_reg = regularize(zp,zmesh)
    return xp_reg, yp_reg, zp_reg

# quadrature extrapolation algorithm
def extrapolation(val_, target_, eligible_):
    val_extpl = np.copy(val_ * eligible_)
    tau_0 = np.copy(target_)
    eps_0 = np.copy(eligible_)
    tau = np.copy(tau_0)
    eps = np.copy(eps_0)
    tau_cur = np.copy(tau)
    eps_cur = np.copy(eps)
    while(np.sum(tau) > 0):
        val_extpl_temp = np.copy(val_extpl)
        for i in range(len(val_)):
            for j in range(len(val_[i])):
                for k in range(len(val_[i,j])):
                    if(tau[i,j,k] == 1):
                        triplet_count = 0
                        triplet_sum = 0
                        #2.9 is used to check if every element in the length-3 array is 1
                        if(np.sum(eps[i+1:i+4,j,k]) > 2.9):
                            triplet_count += 1
                            triplet_sum += 3*val_extpl[i+1,j,k] - 3*val_extpl[i+2,j,k] + val_extpl[i+3,j,k]
                        if(np.sum(eps[i-3:i,j,k]) > 2.9):
                            triplet_count += 1
                            triplet_sum += 3*val_extpl[i-1,j,k] - 3*val_extpl[i-2,j,k] + val_extpl[i-3,j,k]
                        if(np.sum(eps[i,j+1:j+4,k]) > 2.9):
                            triplet_count += 1
                            triplet_sum += 3*val_extpl[i,j+1,k] - 3*val_extpl[i,j+2,k] + val_extpl[i,j+3,k]
                        if(np.sum(eps[i,j-3:j,k]) > 2.9):
                            triplet_count += 1
                            triplet_sum += 3*val_extpl[i,j-1,k] - 3*val_extpl[i,j-2,k] + val_extpl[i,j-3,k]
                        if(np.sum(eps[i,j,k+1:k+4]) > 2.9):
                            triplet_count += 1
                            triplet_sum += 3*val_extpl[i,j,k+1] - 3*val_extpl[i,j,k+2] + val_extpl[i,j,k+3]
                        if(np.sum(eps[i,j,k-3:k]) > 2.9):
                            triplet_count += 1
                            triplet_sum += 3*val_extpl[i,j,k-1] - 3*val_extpl[i,j,k-2] + val_extpl[i,j,k-3]
                            
                        if(triplet_count > 0):
                            val_extpl_temp[i,j,k] = triplet_sum / triplet_count
                            tau_cur[i,j,k] = 0
                            eps_cur[i,j,k] = 1
                        
        tau = np.copy(tau_cur)
        eps = np.copy(eps_cur)
        val_extpl = np.copy(val_extpl_temp)
        
    return val_extpl


# Interpolation method for 3d grid using interpolate from scipy
# mesh (xmesh, ymesh) must be equal-distanced mesh grid
from scipy.interpolate import RegularGridInterpolator
def interpolation_N(mesh, mesh_p, fmesh):
    if(mesh[0].shape == mesh_p[0].shape):
        print("same grid")
        return fmesh
    else:
        return interpolation(mesh,mesh_p,fmesh)

def interpolation(mesh, mesh_p, fmesh):
    xmesh, ymesh, zmesh = mesh
    xmesh_p, ymesh_p, zmesh_p = mesh_p
    x = xmesh[0, :, 0]
    y = ymesh[:, 0, 0]
    z = zmesh[0, 0, :]
    f = RegularGridInterpolator((x, y, z), fmesh)
    fmesh_p = np.zeros_like(fmesh)
    rmesh = np.moveaxis(np.array([xmesh_p,ymesh_p,zmesh_p]), 0, -1)
    
    fmesh_p = f(rmesh)
    fmesh_p = np.moveaxis(fmesh_p,0,1)
    
    return fmesh_p

def no_boundary(phi):
    isIn = mhf3d.get_frame_n(phi)
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    return isIn * (1-N2)

def get_phi(rho_):
    frame_full = mhf3d.get_frame_n(rho_)
    frame_nobnd = no_boundary(rho_)
    d_rho = np.max(rho_ * (frame_full - frame_nobnd))
    print(r"d $\rho / \rho$",d_rho / np.max(rho_))
    return rho_ - d_rho

## poisson solver function
## the result solution is zero (making the mean of the solution 0) at every iteration
# u_init_          : (N*N np array) initial data
# maxIterNum_      : (scalar)       maximum iteration for Jacobi method
# phi_             : (N*N np array) level set
# source_          : (N*N np array) right hand side 
# print_option     : (bool)         switch to print the iteration progress
def poisson_jacobi_solver_zero(u_init_, maxIterNum_, source_, phi_, h,print_option = True):
    u_prev = np.copy(u_init_)
    u      = np.copy(u_init_)
    isIn   = mhf3d.get_frame_n(phi_)
    numIn  = np.sum(isIn)
    for i in range(maxIterNum_):
        # enforce boundary condition
        u[ 0, :, :] = np.zeros_like(u[ 0, :, :])
        u[-1, :, :] = np.zeros_like(u[-1, :, :])
        u[ :, 0, :] = np.zeros_like(u[ :, 0, :])
        u[ :,-1, :] = np.zeros_like(u[ :,-1, :])
        u[ :, :, 0] = np.zeros_like(u[ :, :, 0])
        u[ :, :,-1] = np.zeros_like(u[ :, :,-1])
    
        u_new = np.copy(u)
    
        # update u according to Jacobi method formula
        # https://en.wikipedia.org/wiki/Jacobi_method
        
        del_u = u[1:-1,2:,1:-1] + u[1:-1,0:-2,1:-1] +\
                u[2:,1:-1,1:-1] + u[0:-2,1:-1,1:-1] +\
                u[1:-1,1:-1,2:] + u[1:-1,1:-1,0:-2]
        u_new[1:-1,1:-1,1:-1] = -h**2/6 * (source_[1:-1,1:-1,1:-1] - del_u/h**2)
        u = u_new
        
        # for Neumann condition: normalize the inside to mean = 0
        u -= (np.sum(u*isIn) / numIn)*isIn
        
        # check convergence and print process
        check_convergence_rate = 10**-5
        
        if(i % int(maxIterNum_*0.1) < 0.1):
            u_cur = np.copy(u)
            L2Dif = mhf3d.L_n_norm(np.abs(u_cur - u_prev)) / mhf3d.L_n_norm(u_cur)
            
            if(L2Dif < check_convergence_rate):
                break;
            else:
                u_prev = np.copy(u_cur)
            if(print_option):
                sys.stdout.write("\rJacobi Solver Progress: %4d iterations (max %4d)" % (i,maxIterNum_))
                sys.stdout.flush()
    if(print_option):
        print("")
    
    
    return u



## main coefficient poisson solver function
    
# coefficient Poisson equation:   div ( rho * grad(u)) = rhs
# Neumann boundary condition:     u_n = -grad(u).grad(rho) / |grad(rho)| = boundary
# Level set:                      phi = rho + dh
    
# u_init_          : (N*N np array) initial data
# maxMultiple_     : (scalar)       maximum iteration multiple for Jacobi method
# mesh_            : (duple)        (xmesh, ymesh, zmesh)
# phi_             : (N*N np array) level set
# rhs_             : (N*N np array) right hand side 
# rho_             : (N*N np array) density
# sol_             : (N*N np array) theoretical solution
# boundary_        : (N*N np array) Neumann boundary condition
# iteration_total  : (scalar)       maximum iteration for source term method
def stm_coef_Neumann_3d(u_init_, maxMultiple_, mesh_, phi_, rho_, rhs_, coef_,\
                                       sol_, boundary_, iteration_total):
    
    # making copies of the variables
    phi = np.copy(phi_)
    rho = np.copy(rho_)
    rhs = np.copy(rhs_)
    sol = np.copy(sol_)
    coef = np.copy(coef_)
    boundary = np.copy(boundary_)
    u_cur_result = np.copy(u_init_)
    
    #mesh variables
    xmesh, ymesh, zmesh = mesh_
    h = xmesh[0,1,0] - xmesh[0,0,0]
    N = len(xmesh)
    
    # Level variables
    N1 = get_N1(phi)
    N2 = get_N2(phi)
    Omega_m = mhf3d.D(-phi)
    Omega_p = mhf3d.D(phi)
    isIn = mhf3d.get_frame_n(phi)
    
    #1. Extend g(x,y) off of Gamma, define b throughout N2
    xmesh_p, ymesh_p, zmesh_p = projection((xmesh,ymesh,zmesh), phi_)
    g_ext = interpolation((xmesh, ymesh,zmesh), (xmesh_p, ymesh_p,zmesh_p), boundary)
    a_mesh = g_ext * N2
    x = xmesh[0, :, 0]
    y = ymesh[:, 0, 0]
    z = zmesh[0, 0, :]
    
    #2. extrapolate f throughout N1 U Omega^+
    f_org = np.copy(rhs)
    eligible_0 = Omega_p * (1-N1)
    target_0 = N1 * (1 - eligible_0)
    f_extpl = extrapolation(f_org, target_0, eligible_0)
    
    #3. initialize a based on initial u throughout N2
    u_extpl = extrapolation(u_cur_result, target_0, eligible_0)  
    b_mesh = np.copy(u_extpl)
    
    #4. Find the source term for coefficient
    ux, uy, uz = mhf3d.grad(u_cur_result, h, h, h)
    ux_extpl = extrapolation(ux, target_0, eligible_0)
    uy_extpl = extrapolation(uy, target_0, eligible_0)
    uz_extpl = extrapolation(uz, target_0, eligible_0)
    coefx, coefy, coefz = mhf3d.grad(coef,h,h,h)
    extra = coefx * ux_extpl + coefy * uy_extpl + coefz * uz_extpl
    f_use = (f_extpl - extra) / (coef - singular_null)
    
    # termination array
    Q_array = np.zeros(iteration_total)
    
    # parameters for the iteration
    maxIterNum = maxMultiple_ * N**2
    print_it = True
    for it in range(iteration_total):
            # print iteration process
            if(print_it):
                print("Source term method iteration %d :" % (it + 1))
            
            #A1-1 compute the source term
            source = get_source(a_mesh, b_mesh, phi, f_use, h)
            
            #A1-2 compute the source term with the addition of convergence term
            q = -0.75 * min(1, it*0.1) * 0
            source += (q / h * u_cur_result) * (1-Omega_p) * N2
            #A2 call a Poisson solver resulting in u throughout Omega
            u_result = poisson_jacobi_solver_zero(u_cur_result, maxIterNum, source, phi, h,print_it)
            maxDif,L2Dif = mhf3d.get_error_N(u_result, sol, isIn)
            change = np.abs(u_result - u_cur_result)
            maxChange = np.max(change * isIn)
            
            # Adding relaxation
            rlx = 0.1
            u_cur_result = rlx * u_result + (1 - rlx) * u_cur_result
            
            #A3-1 Extrapolate u throughout N2
            eligible_0 = Omega_p * (1-N1)
            target_0 = N2 * (1-eligible_0)
            u_extpl = extrapolation(u_result, target_0, eligible_0)
            if(it < 0):
                return source
            
            #A3-2 compute the new a throughout N2
            b_mesh = np.copy(u_extpl)
            
            #A3-3 compute the new source term f_use
            ux, uy, uz = mhf3d.grad(u_cur_result, h, h, h)
            ux_extpl = extrapolation(ux, target_0, eligible_0)
            uy_extpl = extrapolation(uy, target_0, eligible_0)
            uz_extpl = extrapolation(uz, target_0, eligible_0)
            extra = coefx * ux_extpl + coefy * uy_extpl + coefz * uz_extpl
            f_use = (f_extpl - extra) / (rho - singular_null)
            
            #A4 check for termination
            Q_array[it] = np.max(u_result * (1-isIn) * N2)
            
            if(it > 5):
                hard_conergence_rate = 10**-4
                hard_convergence = maxChange / (np.max(np.abs(u_extpl)) + mhf3d.singular_null) < hard_conergence_rate
                if(hard_convergence):
                    break
            
    u_result_org = np.copy(u_result)
    
    # Quadruple lagrange extrapolation to the full grid
    isIn_full = mhf3d.get_frame_n(rho)
    eligible_0 = Omega_p * (1-N1)
    target_0 = isIn_full * (1-eligible_0)  
    u_extpl_lagrange = extrapolation(u_result_org, target_0, eligible_0)
    
    return u_extpl_lagrange
