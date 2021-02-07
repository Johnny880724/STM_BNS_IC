# -*- coding: utf-8 -*-
"""
This file generates all the plots used in the paper.
There are five sections in total:
    1. 2D tests without singularity
    2. 2D tests with singular coefficients using constant delta phi
    3. 2D tests with singular coefficients using constant N_sep
    4. N_sep 2D tests
    5. 3D realistic binary neutron star test
    
Created on Thu Jun 25 23:54:37 2020

@author: Johnny Tsao
"""
import numpy as np
import h5py as h5
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
cwd = os.getcwd()
sys.path.append(cwd+"//..//")
import source_term_method_2D as stm2d
import mesh_helper_functions_2D as mhf2d
import mesh_helper_functions_3D as mhf3d
plt.close('all')


def fit(x,y,ends):
    st = ends[0]
    ed = ends[1]
    lx = x[st:ed]
    ly = y[st:ed]
    tempx = lx.reshape((-1,1))
    model = LinearRegression().fit(tempx, ly)
    print("slope: ", "{:.2f}".format(model.coef_[0]), ", from " + str(round(10**x[st])) + " to " + str(round(10**x[ed])))
    return tempx, model

def plot_grid(ax, x,y,h):
    ax.grid(color='gray',alpha = 0.3, linestyle='-', linewidth=1)
    xticks = h*np.linspace(int(np.min(x)/h),int(np.max(x)/h),int(np.max(x)/h)-int(np.min(x)/h)+1, endpoint = True)
    yticks = h*np.linspace(int(np.min(y)/h),int(np.max(y)/h),int(np.max(y)/h)-int(np.min(y)/h)+1, endpoint = True)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    

class plot_obj:
    def __init__(self,file_name,case_str,sigma_str,varname=""):
        self.case_str = case_str
        self.sigma_str = sigma_str
        file_name = file_name
        self.file = h5.File(file_name,"r")
        
        
        val = np.array(self.file.get("sigma_r0_param_eta"),dtype = float)
        self.dh_coef,self.r0,self.param,self.eta = val[0], val[1], val[2], val[3]
        
        self.L2_array = []
        self.L2extpl_array = []
        self.Ngrid_use = []
        plot_err = False
        
        if(plot_err):
                fig_abs = plt.figure()
                ax_abs = fig_abs.gca()
                fig_vepc = plt.figure()
                ax_vepc = fig_vepc.gca()
                fig_rel = plt.figure()
                ax_rel = fig_rel.gca()
        if(len(varname) > 0):
            self.x_array = np.array(self.file.get(varname),dtype =float)
            self.L2_array = np.array(self.file.get("L2"),dtype = float)
            self.L2extpl_array = np.array(self.file.get("L2extpl"),dtype = float)
            self.grid_size = np.array(self.file.get("Ngrid"),dtype =float)
            
        else:
            Ngrid_array = np.array(self.file.get("Ngrid"),dtype =int)
            for i in range(len(Ngrid_array)):
                N_grid = Ngrid_array[i]
                N_half = int(0.5*N_grid)
                vepc = np.array(self.file.get("vepc" + str(i)),dtype = float)
                theory = np.array(self.file.get("theory" + str(i)),dtype = float)
                
                u_init, (xmesh,ymesh), h = stm2d.setup_grid(N_grid)
                r = mhf2d.XYtoR(xmesh,ymesh)
                rho = -(r - self.r0)
                rhoc = self.r0
                
                coef = np.copy(rho)
                phi = rho - self.dh_coef * rhoc
                theory = ymesh + self.param * np.sin(xmesh)
                boundary = (ymesh + self.param * xmesh * np.cos(xmesh)) / (r + mhf2d.singular_null)
                rhs = -((ymesh + self.param * xmesh* np.cos(xmesh)) - self.param*r*(r - self.r0)*np.sin(xmesh))/ (r + mhf2d.singular_null)
                        
                frame = mhf2d.get_frame(phi)
                frame_full =mhf2d.get_frame(rho)
                ux, uy = mhf2d.grad(theory,h,h)
                rhs_theory = mhf2d.div(coef*ux,coef*uy,h,h)
                maxDif, L2Dif = mhf2d.get_error_N(vepc, theory, frame,(False,False))
                # maxDif, L2Dif = mhf2d.get_rel_error_N(vepc, theory, frame,(False,False))
                
                
                
                if(L2Dif < 0.5 and N_grid % 2 ==0):
                    self.Ngrid_use.append(N_grid)
                    self.L2_array.append(L2Dif)
                    maxDif_extpl, L2Dif_extpl = mhf2d.get_error_N(vepc, theory, frame_full,(False,False))
                    self.L2extpl_array.append(L2Dif_extpl)
                    abs_err = np.abs(((vepc-theory)*frame_full))
                    rel_err = abs_err / (np.abs(theory) + mhf2d.singular_null)
                    if(i % 15 == 0 and plot_err):
                        color_Ngrid = (i*0.2 / len(Ngrid_array),i*1.0 / len(Ngrid_array),0.7)
                        ax_abs.plot(xmesh[N_half,:],abs_err[N_half,:], label = "N_grid = " + str(N_grid),color = color_Ngrid)
                        ax_vepc.plot(xmesh[N_half,:], vepc[N_half,:],".",label = "N_grid = " + str(N_grid),color = color_Ngrid)
                        x_rel_d = np.delete(xmesh[N_half,:],[N_half,N_half+1])
                        rel_err_d = np.delete(rel_err[N_half,:],[N_half,N_half+1])
                        ax_rel.plot(x_rel_d, rel_err_d,label = "N_grid = " + str(N_grid),color = color_Ngrid)
                    if(i == len(Ngrid_array) -1 and plot_err):
                        ax_vepc.plot(xmesh[N_half,:], theory[N_half,:])
                elif (N_grid %2 == 1):
                    pass
                else:
                    print("At grid size %d singularity error = %5.0f" % (N_grid, L2Dif))
                if(plot_err):
                    ax_abs.set_xlabel("x")
                    ax_abs.set_ylabel(" abs error")
                    ax_abs.set_title("abs error along x-axis")
                    
                    ax_rel.set_xlabel("x")
                    ax_rel.set_ylabel(" rel error")
                    ax_rel.set_title("rel error along y-axis")
                    
                    ax_vepc.set_xlabel("x")
                    ax_vepc.set_ylabel("vepc")
                    ax_vepc.set_title("vepc along y-axis")
                    
                    ax_rel.legend()
                    ax_abs.legend()
                    ax_vepc.legend()
        self.x = np.log10(self.Ngrid_use)
        # print(self.Ngrid_use)
        self.y1 = np.log10(self.L2extpl_array)
        self.y2 = np.log10(self.L2_array)        
        self.file.close()
                
    def data_convergence_plot(self, plot, out_dir,color_,label,plot_endpoints,fit_endpoints,plot_grid_sw = True):
        fig, ax = plot
        st, ed = plot_endpoints
        x = np.log10(self.Ngrid_use)[st:ed]
        y1 = np.log10(self.L2extpl_array,dtype = float)[st:ed]
        y2 = np.log10(self.L2_array, dtype = float)[st:ed]
        ax.plot(x,y1,label = label + r"$(\vec{x} \in \Omega^+)$",color=color_,fillstyle="full",marker = 'o',alpha = 0.5, markersize = 3, linestyle = "None")
        ax.plot(x,y2,label = label + r"$(\vec{x} \in W^+)$"     ,color=color_,fillstyle='none',marker = 'o',alpha = 0.5, markersize = 3, linestyle = "None")
            
        if(fit_endpoints):
            fit_st,fit_ed = fit_endpoints
            tempx, model = fit(x,y1,(fit_st,fit_ed))
            ax.plot(tempx,model.predict(tempx),'-',label = "slope = " + "{:.2f}".format(model.coef_[0]), linewidth = 2.0, color = color_)
        # model = LinearRegression().fit(tempx, ly1)
        # ax.plot(tempx,model.predict(tempx),'--')
        ymax, ymin = (-2.0, -4.4)
        if(plot_grid_sw):
            plot_grid(ax,x,[ymax,ymin],0.2)
        ax.set_ylim(ymin, ymax)
        # plot_grid(ax,x,np.concatenate((y1,y2)),0.2)
        
        ax.set_xlabel(r"$\log_{10}(N_{grid})$")
        ax.set_ylabel(r"$\log_{10}(E_{2})$")
        # ax.text(2.0,-3.4,"slope = %4g" % model.coef_[0])
        # ax.legend(ncol = 3)
        if(len(out_dir) > 0):
            print("figure saved")
            # plt.savefig(out_dir)
            
    def data_x_y_plot(self, plot, out_dir,color_,title = "",label_="",endpoints=(0,-1)):
        fig, ax = plot
        # x = np.log10(self.x_array)
        # ax.set_xticks(np.log(np.arange(10)))
        # ax.vlines([np.log10(3)],-5,1)
        x =self.x_array
        ax.set_xticks(np.arange(10))
        
        y1 = np.log10(self.L2extpl_array)
        y2 = np.log10(self.L2_array)
        st, ed= endpoints
        st =np.argmax(np.less(y1, -2))
        lx = x[st:ed]
        ly1 = y1[st:ed]
        ly2 = y2[st:ed]
        # ax.plot(x,y1,"b.")
        ax.plot(lx,ly1,".",markersize = 4,color=color_,label = label_)
        ax.plot(lx,ly1,linewidth = 1,color=color_)
        # ax.plot(x,y2,"r.")
        # ax.plot(x,y2,color=color_,label = label_,linestyle = '--')
        
        tempx = lx.reshape((-1,1))
        # model = LinearRegression().fit(tempx, ly1)
        # ax.plot(tempx,model.predict(tempx),'--')
        # ax.set_xlabel(r"$\log_{10}[\delta \phi (*\Delta x)]$")
        ax.set_xlabel(r"$N_{sep}$")
        
        ax.set_ylabel(r"$\log_{10}(E_{2})$")
        
        # ax.text(2.0,-3.4,"slope = %4g" % model.coef_[0])
        ax.legend()
        # ax.set_title("2d singular test case "+self.case_str+" log-log plot " + r'$\sigma = $' + self.sigma_str)
        if(title == ""):
            ax.set_title("2D singular test case "+self.case_str+" sepeartion test")
        else:
            ax.set_title(title)
        if(len(out_dir) > 0):
            print("figure saved")
            # plt.savefig(out_dir)
      

mpl.rcParams.update({'font.size': 8, 'legend.fontsize':8})
figsize_all = (5.4,3.2)
dpi_all = 200
plot_all = [False,False,False,False,False]

# plot_all =[True,True,True,True,True]
# plot_all[0] = True
# plot_all[1]= True
# plot_all[2] = True
# plot_all[3] = True
plot_all[4] = True
#2d Non singular test case plot
#paper as plot_1-1
plot_non_singular = plot_all[0]
if(plot_non_singular):
    cwd = os.getcwd()
    file = h5.File(cwd+"\\2d_not_singular\\test_case_1_dh00_archived.h5","r")
    L2extpl_array = np.array(file.get("L2extpl"),dtype = float)
    Ngrid_array = np.array(file.get("Ngrid"),dtype =float)
    x = np.log10(Ngrid_array)
    y = np.log10(L2extpl_array)
    file.close()
    
    file2 = h5.File(cwd+"\\2d_not_singular\\test_case_1_dh10_archived.h5","r")
    L2extpl_array_dh = np.array(file2.get("L2extpl"),dtype = float)
    Ngrid_array_dh = np.array(file2.get("Ngrid"),dtype =float)
    x_dh = np.log10(Ngrid_array_dh)
    y_dh = np.log10(L2extpl_array_dh)
    file2.close()
    
    fig = plt.figure(figsize=(figsize_all[0],2.8), dpi = dpi_all)
    ax = fig.gca()
    
    tempx, model = fit(x,y,(0,44))
    
    
    
    # ax.plot(x_dh,y_dh)
    # ax.plot(x,y)
    tempx_sep, model_sep = fit(x_dh,y_dh,(35,53))
    
    plot_grid(ax,x,y,0.2)
    
    ax.plot(tempx,model.predict(tempx)            ,'-', label = "slope = " + "{:.2f}".format(model.coef_[0]), linewidth = 2.5)
    ax.plot(tempx_sep,model_sep.predict(tempx_sep),'-', label = "slope = " + "{:.2f}".format(model_sep.coef_[0]), linewidth = 2.5)
    ax.plot(x_dh,y_dh,"o",label = r"$\delta \varphi = 0.10 \rho_0$",color = "red" ,alpha= 0.5,markersize = 3)
    ax.plot(x,y      ,"o",label = r"$\delta \varphi = 0$"          ,color = "blue", alpha = 0.5,markersize = 3)
    
    ax.set_xlabel(r"$\log_{10}(N_{grid})$")
    ax.set_ylabel(r"$\log_{10}(E_{2})$")
    
    
    plt.title("2D non-singular test: case 1")
    plt.legend()
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.20)
    fig.subplots_adjust(top=0.90)
    fig.subplots_adjust(right=0.95)

    fig.savefig("plot_1-1.pdf")

            
## test case with constant sigma
# paper plot_2-1, plot_2-2
plot_const_sigma_all = plot_all[1]
if(plot_const_sigma_all):
    case_str_list = ["2","3"]
    # case_str_list = ["2"]
    i=0
    # ed = {"2": (19,17), "3": (17,15)}
    ed = {"2": (42,45), "3": (34,24)}
    legend_loc = {"2": (0.2,0.75), "3": (0.2,0.75)}
    
    if(False):
        file_name_1 = cwd+"\\2d_singular\\const_delta_phi\\test_case_1_dh05.h5"
        file_name_2 = cwd+"\\2d_singular\\const_delta_phi\\test_case_1_dh10.h5"  
        myPlot_1 = plot_obj(file_name_1,"1","05")
        myPlot_2 = plot_obj(file_name_2,"1","10")
        fig = plt.figure(figsize = figsize_all, dpi = dpi_all)
        ax1 = fig.gca()
        ax2 = fig.gca()
        myPlot_1.data_convergence_plot((fig,ax1),"","blue",r"$\delta \varphi = 0.05\rho_0$",(0,10))
        myPlot_2.data_convergence_plot((fig,ax2),"","red",r"$\delta \varphi = 0.10\rho_0$",(0,10))
        plt.ylim(-4.4, -3.0)
        plt.title("2D singular test : case 2.1" + r" with const $\delta \varphi$")
        ax1.legend(ncol = 2,labelspacing = 0.0)
    
    # test case 2-1
        
    sigma_str_1 = "05"
    sigma_str_2 = "10"
    file_name_1 = cwd+"\\2d_singular\\const_delta_phi\\test_case_2_dh05_archived.h5"
    file_name_2 = cwd+"\\2d_singular\\const_delta_phi\\test_case_2_dh10_archived.h5"  
    myPlot_1 = plot_obj(file_name_1,"1","05")
    myPlot_2 = plot_obj(file_name_2,"1","10")
    fig = plt.figure(figsize = figsize_all, dpi = dpi_all)
    ax1 = fig.gca()
    ax2 = fig.gca()
    myPlot_1.data_convergence_plot((fig,ax1),"","blue",r"$\delta \varphi = 0.05\rho_0$",(0,47),(0,42))
    myPlot_2.data_convergence_plot((fig,ax2),"","red",r"$\delta \varphi = 0.10\rho_0$",(0,47),(0,45))
    # print(myPlot_1.Ngrid_use)
    plt.title("2D singular test : case 2.1" + r" with const $\delta \varphi$")
    # ax1.legend(loc = (0.24,0.75) ,ncol = 2,labelspacing = 0.0)
    ax1.legend(loc = (0.635,0.58) ,ncol = 1,labelspacing = 0.0)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.20)
    fig.subplots_adjust(top=0.90)
    fig.subplots_adjust(right=0.95)
    plt.xlim(1.48,2.28)
    plt.xlim(1.48,2.29)
    plt.ylim(-4.5,-2.1)
    plt.savefig("plot_2-1.pdf")
           
    
    file_name_1 = cwd+"\\2d_singular\\const_delta_phi\\test_case_3_dh05_archived.h5"
    file_name_2 = cwd+"\\2d_singular\\const_delta_phi\\test_case_3_dh10_archived.h5"  
    myPlot_1 = plot_obj(file_name_1,"2","05")
    myPlot_2 = plot_obj(file_name_2,"2","10")
    fig = plt.figure(figsize = figsize_all, dpi = dpi_all)
    ax1 = fig.gca()
    ax2 = fig.gca()
    myPlot_1.data_convergence_plot((fig,ax1),"","blue",r"$\delta \varphi = 0.05\rho_0$",(0,45),(0,34))
    myPlot_2.data_convergence_plot((fig,ax2),"","red",r"$\delta \varphi = 0.10\rho_0$",(0,28),(0,24),False)
    
    
    # print(myPlot_1.Ngrid_use)
    plt.title("2D singular test : case 2.2" + r" with const $\delta \varphi$")
    ax1.legend(loc = (0.635,0.58) ,ncol = 1,labelspacing = 0.0)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.20)
    fig.subplots_adjust(top=0.90)
    fig.subplots_adjust(right=0.95)
    plt.xlim(1.48,2.29)
    plt.ylim(-4.4,-2.7)
    plt.savefig("plot_2-2.pdf")
           
## test case with varing sigma
# paper plot_2-3
plot_vary_sigma2 = plot_all[2]
if(plot_vary_sigma2):
    test_case_list = ["2"]
    
    file_name_1 = cwd+"\\2d_singular\\const_N\\test_case_2_N2_archived.h5"
    myPlot_1 = plot_obj(file_name_1,"2","05")
    file_name_3 = cwd+"\\2d_singular\\const_N\\test_case_2_N4_archived.h5"
    myPlot_3 = plot_obj(file_name_3,"2","20")
    
    fig = plt.figure(figsize = figsize_all, dpi = dpi_all)
    ax1 = fig.gca()
    ax3 = fig.gca()
    
    myPlot_1.L2extpl_array = np.delete(myPlot_1.L2extpl_array,31)
    myPlot_1.L2_array = np.delete(myPlot_1.L2_array,31)
    myPlot_1.Ngrid_use = np.delete(myPlot_1.Ngrid_use,31)
    
    myPlot_1.data_convergence_plot((fig,ax1),"","blue",r"$N_{sep} = 2.0$ ",(0,-1),(0,15))
    # myPlot_1.data_convergence_plot((fig,ax1),"","blue",r"$N_{sep} = 3.0$ ",(0,-1),(0,24))
    
    
    myPlot_3.L2extpl_array = np.delete(myPlot_3.L2extpl_array,31)
    myPlot_3.L2_array = np.delete(myPlot_3.L2_array,31)
    myPlot_3.Ngrid_use = np.delete(myPlot_3.Ngrid_use,31)
    # myPlot_2.data_convergence_plot((fig,ax2),"","green",r"$\delta\varphi = 3.0 \Delta x$ ")
    myPlot_3.data_convergence_plot((fig,ax3),"","red",r"$N_{sep} = 4.0$ ",(0,-1), (0,33))

    plt.title("2D singular test: case 2.1 with const "+r"$N_{sep}$")
    # plt.legend(loc = (0.25,0.75), ncol = 2,labelspacing = 0.0)
    plt.legend(loc = (0.65,0.58), ncol = 1,labelspacing = 0.0)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.20)
    fig.subplots_adjust(top=0.90)
    fig.subplots_adjust(right=0.95)
    plt.xlim(1.48, 2.2)
    plt.ylim(-4.5,-2.4)
    fig.savefig("plot_2-3.pdf")
        
## test case with varing sigma
# used in paper plot_2-sigma
plot_sigma_test = plot_all[3]
if(plot_sigma_test):
    test_case = "2"
    # grid_size_array =[32,64,96,128]
    grid_size_array =[32,64,128]
    # grid_size_array = [128]
    color = ["blue","green","red"]
    # color = ["blue","green","red","yellow"]
    fig = plt.figure(figsize = figsize_all, dpi = dpi_all)
    for i in range(len(grid_size_array)):
        grid_size = grid_size_array[i]
        file_name_1 = cwd+"\\2d_singular\\N_test\\test_case_2_grid"+"{:2}".format(grid_size)+".h5"
        myPlot_1 = plot_obj(file_name_1,test_case,"05","num_grid_dr")
        ax1 = fig.gca()
        myPlot_1.data_x_y_plot((fig,ax1),"",color[i],"",r"$N_{grid} = $"+str(grid_size),(2,-1))
        ax1.vlines([1,3],-5,1,linestyle = ':')
        # ax1.vlines([2],-5,1, color='grey', lw=16.8*figsize_all[0], alpha=0.1)
        ax1.vlines([2],-5,1, color='grey', lw=22.8*figsize_all[0], alpha=0.1)
        ax1.set_ylim(-5,-2)
    plt.title("Varying separation: case 2.1")
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.20)
    fig.subplots_adjust(top=0.90)
    fig.subplots_adjust(right=0.95)
    fig.savefig("plot_2-sigma.pdf")
        
        
# paper plot_3
bns_3d = plot_all[4]
if(bns_3d):
    R0 = 1.218687320433609E+01 # length normalization constant
    # file = h5.File(cwd+"\\3d_bns\\bns3d_result_dhnone" + ".h5","r")
    file = h5.File(cwd+"\\3d_bns\\bns3d_result_final_archived" + ".h5","r")
    vepc = np.array(file.get("vepc"),dtype = float)
    theory = np.array(file.get("theory"),dtype = float)
    rho = np.array(file.get("rho"),dtype = float)
    vxtheory = np.array(file.get("vxtheory"),dtype = float)
    vytheory = np.array(file.get("vytheory"),dtype = float)
    vztheory = np.array(file.get("vztheory"),dtype = float)
    xmesh = np.array(file.get("x"),dtype = float)
    ymesh = np.array(file.get("y"),dtype = float)
    zmesh = np.array(file.get("z"),dtype = float)
    file.close()
    xplot = xmesh*R0
    yplot = ymesh*R0
    zplot = zmesh*R0
    
    dx = xmesh[0,1,0] - xmesh[0,0,0]
    dy = ymesh[1,0,0] - ymesh[0,0,0]
    dz = zmesh[0,0,1] - zmesh[0,0,0]
    
    N = len(vepc)
    layer = int(0.5*N)
    isIn_large = mhf3d.get_frame(rho)
    abs_dif = np.abs(vepc - theory)*isIn_large
    abs_dif[layer,:,:] = np.zeros_like(abs_dif[layer,:,:])
    theory_noSing = np.copy(theory)
    print("maximum difference is " , np.max(abs_dif))
    print("maximum rel difference is " , np.max(np.abs(abs_dif / (theory_noSing + mhf3d.singular_null))))
    
    mpl.rcParams.update({'font.size': 6, 'legend.fontsize':6})
    
    plot_3d = True
    if(plot_3d):
        plot_frame = isIn_large*1.0
        def plot2d_compare(mat1, mat2, *arg):
            from matplotlib import cm
            frame = np.ones_like(mat1)
            this_cmap = cm.coolwarm
            if(arg):
                frame = arg[0]
                
            from matplotlib import ticker
            from matplotlib.ticker import FormatStrFormatter
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.format = '%0.2f'
            formatter.set_scientific(True) 
            formatter.set_powerlimits((0,0)) 
            
            
            # figdim = 2.0
            # f_x = 2; f_y =2
            # fig, ax = plt.subplots(f_x,f_y,figsize=(f_y*figdim,f_x*figdim))
            # # fig = plt.figure(figsize=(f_y*figdim,f_x*figdim))
            
            # plt.tight_layout()
            # fig.subplots_adjust(left=0.01)
            # fig.subplots_adjust(bottom=0.05)
            # fig.subplots_adjust(top=0.90)
            # fig.subplots_adjust(right=0.99)
            
            
            # # print(ax)
            
            
            # pt1 = ax[int(0/f_y),0%f_y].matshow(mat1*frame,cmap = this_cmap)
            # ax[int(0/f_y),0%f_y].set_title("STM result")
            # fig.colorbar(pt1,ax = ax[int(0/f_y),0%f_y], format=formatter)
            
            
            # pt2 = ax[int(1/f_y),1%f_y].matshow(mat2*frame,cmap = this_cmap)
            # ax[int(1/f_y),1%f_y].set_title("COCAL")
            # fig.colorbar(pt2,ax = ax[int(1/f_y),1%f_y], format=formatter)
            
            # pt3 = ax[int(2/f_y),2%f_y].matshow(np.abs(mat1 - mat2)*frame,cmap=this_cmap)
            # ax[int(2/f_y),2%f_y].set_title("abs dif")
            # fig.colorbar(pt3,ax = ax[int(2/f_y),2%f_y], format=formatter)
            
            # frame_small = mhf3d.get_frame(np.abs(frame*mat2) - 1e-15*np.max(np.abs(frame*mat2)))
            # rel_err = np.abs(mat1 - mat2)/ (np.abs(mat2) + mhf3d.singular_null)
            # frame_min = np.min(frame_small * rel_err)
            # frame_max = np.max(frame_small * rel_err)
            # pt4 = ax[int(3/f_y),3%f_y].matshow(rel_err * frame,cmap=this_cmap, vmin = frame_min, vmax = frame_max)
            # ax[int(3/f_y),3%f_y].set_title("relative difference")
            # fig.colorbar(pt4,ax = ax[int(3/f_y),3%f_y], format=formatter)
            
            # if(len(arg) > 1):
            #     fig.suptitle(arg[1])
            
            # for i in range(2):
            #     for j in range(2):
            #         ax[i,j].xaxis.set_ticks_position('bottom')
            #         ax[i,j].yaxis.set_ticks_position('left')
            #         if(len(arg) > 2):
            #             ax[i,j].set_xlabel(arg[2][0])
            #             ax[i,j].set_ylabel(arg[2][1])
                        
                        
                        
                        
            fig = plt.figure(figsize = (2.7,2.7), dpi = 400)
            fig.suptitle(arg[1])
            sub_w = 0.30
            sub_h = 0.30
            ax =[ plt.axes([0.15, 0.55, sub_w, sub_h]),\
                  plt.axes([0.62, 0.55, sub_w, sub_h]),\
                  plt.axes([0.15, 0.1, sub_w, sub_h]),\
                  plt.axes([0.62, 0.1, sub_w, sub_h])]
            pt1 = ax[0].matshow(mat1*frame,cmap = this_cmap)
            ax[0].set_title("STM result")
            cb1 = fig.colorbar(pt1,ax = ax[0], format=formatter)
            cb1.ax.yaxis.set_offset_position('left')    
            
            pt2 = ax[1].matshow(mat2*frame,cmap = this_cmap)
            ax[1].set_title("COCAL")
            cb2 = fig.colorbar(pt2,ax = ax[1], format=formatter)
            cb2.ax.yaxis.set_offset_position('right')  
            
            pt3 = ax[2].matshow(np.abs(mat1 - mat2)*frame,cmap=this_cmap)
            ax[2].set_title("abs dif")
            cb3 = fig.colorbar(pt3,ax = ax[2], format=formatter)
            cb3.ax.yaxis.set_offset_position('left')  
            
            frame_small = mhf3d.get_frame(np.abs(frame*mat2) - 1e-15*np.max(np.abs(frame*mat2)))
            rel_err = np.abs(mat1 - mat2)/ (np.abs(mat2) + mhf3d.singular_null)
            frame_min = np.min(frame_small * rel_err)
            frame_max = np.max(frame_small * rel_err)
            pt4 = ax[3].matshow(rel_err * frame,cmap=this_cmap, vmin = frame_min, vmax = frame_max)
            ax[3].set_title("rel dif")
            
            cb4 = fig.colorbar(pt4,ax = ax[3], format=formatter)
            cb4.ax.yaxis.set_offset_position('left')  
            for i in range(4):
                ax[i].xaxis.set_ticks_position('bottom')
                ax[i].yaxis.set_ticks_position('left')
                ax[i].set_xticks([0,20,40,60])
                ax[i].set_yticks([0,20,40,60])
                ax[i].set_xlabel(arg[2][0])
                ax[i].set_ylabel(arg[2][1])
                ax[i].xaxis.labelpad = 0.5
                ax[i].yaxis.labelpad = 0.5
                # plt.tight_layout()
            fig.subplots_adjust(left=0.05)
            fig.subplots_adjust(bottom=0.20)
            fig.subplots_adjust(top=0.90)
            fig.subplots_adjust(right=0.95)
            
            return fig
            
            
        # for i in range(2):
        fig1 = plot2d_compare(vepc[:,:,layer], theory[:,:,layer],plot_frame[:,:,layer],"velocity potential comparison (x-y plane)",["x grid","y grid"])
        fig2 = plot2d_compare(vepc[:,layer,:], theory[:,layer,:],plot_frame[:,layer,:],"velocity potential comparison (y-z plane)",["z grid","y grid"])
        
        # plot2d_compare(vepc[layer,:,:], theory[layer,:,:],plot_frame[layer,:,:],"velocity potential comparison (z-x plane)",["z grid","x grid"])
        fig1.savefig("plot_3-1-xy.pdf")
        fig2.savefig("plot_3-1-yz.pdf")
    # 
    def adjust(fig):
        fig.subplots_adjust(left=0.10)
        fig.subplots_adjust(bottom=0.20)
        fig.subplots_adjust(top=0.90)
        fig.subplots_adjust(right=0.99)
        plt.tight_layout()
        
    mpl.rcParams.update({'font.size': 8, 'legend.fontsize':6})
    plot_v = True
    if(plot_v):
        vxi_result, vyi_result, vzi_result = mhf3d.grad(vepc,dx,dy,dz)
                
        vxi_result, vyi_result, vzi_result = mhf3d.grad(vepc,dx,dy,dz)
        figsize_plot = (1.8,2.0)
        fig = plt.figure(figsize = figsize_plot, dpi = 300)
        ax = fig.gca()
        
        ax.plot(xplot[layer,:,layer],vytheory[layer,:,layer], label = "COCAL")
        ax.plot(xplot[layer,:,layer],vyi_result[layer,:,layer],label = "STM")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("$\partial_y \Phi$")   
        # ax.set_title("$\partial_y \Phi$ along x-axis")
        
        adjust(fig)
        fig.savefig("plot_3-2-vy-x.pdf")
        
        print("vy-x rel", np.max(np.abs(vytheory[layer,:,layer] - vyi_result[layer,:,layer])) / (np.max(np.abs(vytheory[layer,:,layer]) + 1e-10)))
        
        fig = plt.figure(figsize = figsize_plot , dpi = 300)
        ax = fig.gca()
        
        ax.plot(zplot[layer,layer,:],vytheory[layer,layer,:], label = "COCAL")
        ax.plot(zplot[layer,layer,:],vyi_result[layer,layer,:],label = "STM")
        ax.legend()
        ax.set_xlabel("z")
        ax.set_ylabel("$\partial_y \Phi$")   
        # ax.set_title("$\partial_y \Phi$ along  z-axis")
        adjust(fig)
        fig.savefig("plot_3-2-vy-z.pdf")
        print("vy-z rel", np.max(np.abs(vytheory[layer,layer,:] - vyi_result[layer,layer,:])) / (np.max(np.abs(vytheory[layer,layer,:]) + 1e-10)))
        
        fig = plt.figure(figsize = figsize_plot, dpi = 300 )
        ax = fig.gca()
        
        ax.plot(yplot[:,layer,layer],vxtheory[:,layer,layer], label = "COCAL")
        ax.plot(yplot[:,layer,layer],vxi_result[:,layer,layer],label = "STM")
        ax.legend()
        ax.set_xlabel("y")
        ax.set_ylabel("$\partial_x \Phi$")   
        # ax.set_title("$\partial_x \Phi$ along y-axis")
        
        adjust(fig)
        fig.savefig("plot_3-2-vx-y.pdf")
        
        
        
        print("vx-y rel", np.max(np.abs(vxtheory[:,layer,layer] - vxi_result[:,layer,layer])) / (np.max(np.abs(vxtheory[:,layer,layer]) + 1e-10)))
