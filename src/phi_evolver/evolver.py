
import numpy as np, os, pandas as pd
from astropy import units as u
from astropy import constants as c

from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp


# constant
eV2Mpc = (1*u.eV/(c.hbar*c.c)).to(1/u.Mpc).value


def compute_evolution(dir_evolver, dir_birefclass, 
                      logm, n=1, tf=100, xi=None, eps=1e-3, z_start=60,z_end=50, 
                      phi0_ini = 1, phi1_ini = 0,
                      h=0.6766, omega_b=0.02242, omega_cdm=0.11933, Omega_k=0., T_cmb=2.7255,
                      dir_output=None, normalized=True,
                     ):
    '''
    Main function to compute phi(a)
    '''

    # mass
    mass = 10**(logm)*eV2Mpc

    # compute background quantities and load a, H(a), dH(a)/da
    a, Ha, dHada = compute_background(dir_evolver,dir_birefclass,
                       h=h,omega_b=omega_b,omega_cdm=omega_cdm,Omega_k=Omega_k,T_cmb=T_cmb,
                       dir_output=dir_output,
                      )

    # solution
    x_a = solve_ivp(EoM_phi, [a[0],1], [phi0_ini,phi1_ini], 
                    t_eval=a, args = [mass, tf, n, Ha, dHada, xi, eps, z_start, z_end]
                   )

    phi  = x_a.y[0]
    phi0 = phi[-1]

    # compute (phi(0)-phi(z))/(phi(0)-1) = beta(z)/beta_0
    if normalized:
        phi = (phi-phi0)/(1.-phi0)
    
    return a, phi, Ha, dHada



def compute_background(dir_evolver,dir_birefclass,
                       h=0.6766, omega_b=0.02242, omega_cdm=0.11933, Omega_k=0., T_cmb=2.7255,
                       dir_output=None,
                      ):

    dir_class_aux = dir_evolver + 'class_aux/'

    if dir_output is None:
        dir_class_output = dir_evolver + 'class_output/'
    else:
        dir_class_output = dir_output
    
    file_class_ini_org = dir_class_aux + 'background.ini'
    file_class_ini_new = dir_class_output + 'background.ini'
    file_class_message = dir_class_output + 'test.log'

    # create inifile for CLASS
    edit_inifile(file_class_ini_org, file_class_ini_new, dir_class_output, 
                 h=h, omega_b=omega_b, omega_cdm=omega_cdm,
                )

    # run CLASS
    os.system(dir_birefclass+'/class '+file_class_ini_new+' > '+file_class_message)

    # load a, H(a), dH(a)/da
    a, Ha, dHada = load_background(dir_class_output+'/background.dat')

    return a, Ha, dHada
    


def edit_inifile(inifile_org, inifile_new, output_data_path,
                h=0.6766, omega_b=0.02242, omega_cdm=0.11933, Omega_k=0., T_cmb=2.7255):
    '''
    Edit CLASS inifile to compute background quantities
    '''
    
    # reading original file
    with open(inifile_org, 'r') as f:
        lines = f.readlines()

    with open(inifile_new, 'w') as g:

        # add the following lines
        config_text = (
            f"filename_axion_dynamics = \n"
            f"length_resolution_axion_dynamics = \n"
            f"isotropic_alpha_0 = \n"
            f"is_m_minus = \n"
            f"root = {output_data_path}\n"
            f"h = {h}\n"
            f"omega_b = {omega_b}\n"
            f"omega_cdm = {omega_cdm}\n"
            f"Omega_k = {Omega_k}\n"
            f"T_cmb = {T_cmb}\n"
        )
        g.write(config_text)
    
        # other parts
        for line in lines:
            g.write(line)



def load_background(filename):
    '''
    load background file and return a, H(a) and dH(a)/da
    '''

    # load pre-computed background
    df_bg = pd.read_table(filename,comment='#',header=None,sep=r'\s+')
    df_bg = df_bg.rename(columns={0: 'z', 1:'proper time [Gyr]', 2:'conf.time [Mpc]',  3:'H [1/Mpc]',4:'comov.dist.', 5:'ang.diam.dist.', 6:'lum.dist.', 7:'comov.snd.hrz.', 8:'rho_g', 9:'rho_b', 10:'rho_cdm', 11:'rho_lambda', 12:'rho_ur', 13:'rho_crit', 14:'rho_tot', 15:'p_tot', 16:'p_tot_prime', 17:'gr.fac. D', 18:'gr.fac. f'})

    # scale factor
    a = df_bg['comov.dist.']/df_bg['lum.dist.']
    a = a.fillna(1.0)

    # H(a)
    Ha = CubicSpline( a, df_bg['H [1/Mpc]'])

    # dH(a)/da
    dHada = Ha.derivative()

    return a, Ha, dHada
    

# phi parameters
def params_phi(model='ALP'):
    '''
    n      : n=1 for ALP and n=2 for EDE potential
    tf     : f_a/phi_ini; the symmetry breaking scale normalized by phi_ini, dimensionless
    '''
    if model=='ALP':
        n, tf = 1, 100.
    if model=='EDE':
        n, tf = 2, np.sqrt(8*np.pi)
    return n, tf


def V_phi(x,mass,tf,n,deriv=0):
    '''
    x      : pseudoscalar field normalized by phi_ini, dimensionless
    mass   : the mass parameter in the unit of eV
    tf     : the symmetry breaking scale normalized by phi_ini, dimensionless
    deriv  : the order of derivative with respect to phi
    '''
    
    if deriv == 0: # potential
        return mass**2*tf**2*(1.-np.cos(x/tf))**n

    if deriv == 1: # dV/dphi
        return n*mass**2*tf*(1.-np.cos(x/tf))**(n-1)*np.sin(x/tf)

    if deriv == 2: # d^2V/dphi^2
        if n == 1:
            return mass**2*np.cos(x/tf)
        else:
            return n*mass**2*(1.-np.cos(x/tf))**(n-2)*(-1.+np.cos(x/tf)+n*np.sin(x/tf)**2)
            


def EoM_phi(a, x, mass, tf, n, Ha, dHada, xi=None, eps=1e-3, z_start=60,z_end=50):
    
    '''
    # Solving the following equations
    #   dx0/da = x1
    #   dx1/da = d^2x0/da^2 = - ( 4/a + dlnH/da ) dx0/da + (dV/dphi)/(a^2H^2)
    # where x0 is varphi=phi/phi_ini and x1 is d(varphi)/da. 
    # If V = m^2 x0^2 / 2, the above second equation becomes
    #   dx1/da = d^2x0/da^2 = - ( 4/a + dlnH/da ) dx0/da + (m/H0)^2 x0 / (a^2E^2)
    '''

    # conformal expansion rate
    calH = a*Ha(a)

    # redshift
    z = 1./a - 1.
    
    if xi is not None and z>z_end and z<z_start:
        frac = 1e-3 # energy density fraction of ALP to matter
        Q    = (xi/a) * (0.5*mass**2/frac/a**3)
        Src  = Q * x[1] / (x[1]**2 + eps**2)
    else:
        Src = 0.
    
    return [ x[1], - ( (4./a+dHada(a)/Ha(a))*x[1] + ( V_phi(x[0],mass,tf,n,deriv=1) - Src ) / calH**2 ) ]
    


#////////////////////////////////////////////////////////////////////////////////////////////////////#
# not used in the main function, just used for cross-checking the results
#////////////////////////////////////////////////////////////////////////////////////////////////////#

def E_a(a,Om,zeq=3402):
    '''
    H(a)/H0
    '''
    Or = Om * 1./(1.+zeq)
    return np.sqrt( Om/a**3 + Or/a**4 + 1-Om)


def dlnEa_da(a,Om,zeq=3402):
    '''
    dH(a)/da/H(a)
    '''
    Or = Om * 1./(1.+zeq)
    return -(1.5*a*Om+2*Or)/(a**5*E_a(a,Om)**2)
    

