import scipy.constants as constants
import numpy as np
import healpy as hp
import pysm3.units as u
from tqdm import tqdm
import pymaster as nmt
import scipy.stats
import sympy as sym

#SEDs:

def B(nu, b_T):
    """Planck function.

    Parameters
    ----------
    :param nu: frequency in GHz at which to evaluate planck function.
    :type nu: float.
    :param b_T: inverse temperature of black body.
    :type b_T: float.

    Returns
    -------
    :return: float -- black body brightness.
    
    """
    x = constants.h*nu*1.e9*b_T/constants.k
    return 2.*constants.h *(nu *1.e9)**3/ constants.c**2/np.expm1(x)

def mbb(nu,beta,b_T):
    """Modified blackbody function.

    :param nu: frequency in GHz at which to evaluate the SED.
    :type nu: float.
    :param beta: spectral index of modified blackbody.
    :param b_T: inverse temperature of modified black body.
    :type b_T: float.
    :return: float -- modified black body brightness.
    
    """
    return B(nu,b_T)*(1e9*nu)**beta

def MBBpysm(freq,A,beta,b_T,nu0):
    """Modified blackbody function to reproduce Pysm models.

    :param freq: array of frequencies in GHz at which to evaluate the SED.
    :type freq: array of floats.
    :param A: Amplitude map for the model in muK_CMB
    :type A: array of floats.
    :param beta: spectral index of modified blackbody.
    :param b_T: inverse temperature of modified black body.
    :type b_T: float.
    :param nu0: pivot frequency in GHz.
    :type nu0: float.        
    :return: float -- modified black body brightness.
    
    """
    factor= u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))/u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0*u.GHz))
    mapd=np.array([A*mbb(freq[f],beta-2,b_T)/mbb(nu0,beta-2,b_T)*factor[f] for f in range(len(freq))])
    return mapd

def PLpysm(freq,A,beta,nu0):
    """Power-law function to reproduce Pysm models.

    :param freq: array of frequencies in GHz at which to evaluate the SED.
    :type freq: array of floats.
    :param A: Amplitude map for the model in muK_CMB
    :type A: array of floats.
    :param beta: spectral index of the power-law.
    :param nu0: pivot frequency in GHz.
    :type nu0: float.
    :return: float -- power-law brightness.
    
    """
    factor= u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))/u.K_RJ.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0*u.GHz))
    mapd=np.array([A*(freq[f]/nu0)**(beta)*factor[f] for f in range(len(freq))])
    return mapd


#general map operations:

def downgrade_alm(input_alm,nside_in,nside_out):
    """
    This is a Function to downgrade Alm correctly.
    nside_in must be bigger than nside_out.
    In this function, lmax_in = 3*nside_in-1 , lmax_out = 3*nside_out-1 .
    input_alm must be lmax = lmax_in and output_alm must be lmax = lmax_out.
    This function get only values in the range 0 < l < lmax_out from input_alm,
    and put these values into output_alm which has range 0 < l < lmax_out.
    """
    lmax_in = nside_in*3-1
    lmax_out = nside_out*3-1
    output_alm = np.zeros((3,hp.sphtfunc.Alm.getsize(lmax_out)),dtype='complex128')
    
    for m in range(lmax_out+1):
        idx_1_in = hp.sphtfunc.Alm.getidx(lmax_in,m ,m)
        idx_2_in = hp.sphtfunc.Alm.getidx(lmax_in,lmax_out ,m)

        idx_1_out = hp.sphtfunc.Alm.getidx(lmax_out,m ,m)
        idx_2_out = hp.sphtfunc.Alm.getidx(lmax_out,lmax_out ,m)

        output_alm[:,idx_1_out:idx_2_out+1] = input_alm[:,idx_1_in:idx_2_in+1]
    return output_alm

def downgrade_map(input_map,nside_out,nside_in=512):
    """
    This is a Function to downgrade map correctly in harmonic space.
    nside_in must be bigger than nside_out.
    input_map must have nside_in.
    output_map has nside_out as Nside
    """
    #  nside_in= hp.npix2nside(len(input_map))
    if nside_out==nside_in:
        return input_map
    else:
        input_alm = hp.map2alm(input_map)  #input map → input alm
        output_alm = downgrade_alm(input_alm,nside_in,nside_out) # input alm → output alm (decrease nside)
        output_map = hp.alm2map(output_alm,nside=nside_out)#  output alm → output map
        return output_map

#Power spectra functions

def compute_master(f_a, f_b, wsp):
    """compute decoupled CL with a workspace"""
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

def compute_cl(mapd,mask,b):
    """compute simple CL"""
    fa1 = nmt.NmtField(mask, (mapd)*1,purify_e=False, purify_b=True)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(fa1, fa1, b)
    return compute_master(fa1,fa1,wsp)        

def compute_cross_cl(mapd1,mapd2,mask,b):
    """compute cross angular power-spectra"""
    fa1 = nmt.NmtField(mask, (mapd1)*1,purify_e=False, purify_b=True)
    fa2 = nmt.NmtField(mask, (mapd2)*1,purify_e=False, purify_b=True)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(fa1, fa2, b)
    return compute_master(fa1,fa2,wsp)      

def decorr(mapd,F,mask,b,mode='BB'):
    """spectral decorrelation"""
    sp_dict = {'EE': 0, 'EB': 1, 'BE':2, 'BB': 3}
    sp = sp_dict.get(mode, None)
    clF= compute_cl(mapd[F],mask,b)
    clF_1= compute_cl(mapd[F-1],mask,b)
    crosscl= compute_cross_cl(mapd[F],mapd[F-1],mask,b)
    return crosscl[sp,2:]/np.sqrt(clF[sp,2:]*clF_1[sp,2:])

def rEB(mapd,F,mask,b):
    """E/B ratio dependence with frequency """
    clF= compute_cl(mapd[F],mask,b)
    return clF[0,2:]/clF[3,2:]

#Moments computation 

#maximal values for the pivots in order to avoid divergences when depolarisation is too strong:
betarange= [0.9,2.5]
b_Trange= [1/35,1/10]

def crop_outliers(mom,maxborder=4,maxtorder=4,nsig=10):
    """
    Supress possible outliers in moment maps to avoid possible numerical issues.
    """
    for b in range(maxtorder+1):
        for t in range(maxborder+1):
            ipixI= np.where(abs(mom[0,b,t])>=np.mean(abs(mom[0,b,t])+nsig*np.std(abs(mom[0,b,t]))))[0] 
            ipixP= np.where(abs(mom[1,b,t])>=np.mean(abs(mom[1,b,t])+nsig*np.std(abs(mom[1,b,t]))))[0] 

            mom[0,b,t,ipixI]= 0.0+0.0*1j
            mom[1,b,t,ipixP]= 0.0+0.0*1j
    mom[0,0,0]=np.ones(len(mom[0,0,0]))
    mom[1,0,0]=np.ones(len(mom[0,0,0]))
    return mom
    
def compute_mom(nside,modelnu0,betamap,tempmap,maxborder=3,maxtorder=3,SED_type='mbb',betabar=None,tempbar=None, pivot_type='2D'):
    """
    Compute the moment value in each pixel

    Parameters
    ----------
    nside: int
        Resolution parameter at which this model is to be calculated.
    modelnu0: triplet of healpy map of size npix
        Template of (I,Q,U) for the model at reference frequency nu0
    betamap: numpy array of size (nlayer,npix)
        Map of the beta values contained in each pixel. For two dimensional models (e.g. d1, d10...), must be of size (1,npix).
    tempmap: numpy array of size (nlayer,npix)
        Map of the temperature values contained in each pixel. For two dimensional models (e.g. d1, d10...), must be of size (1,npix).
    maxborder: float
        Maximum order of the expansion in beta at which the moments must be computed
    maxtorder: float
        Maximum order of the expansion in temp at which the moments must be computed
    SED_type: string
        Type of SED, either 'mbb' or 'pl'
    betabar: float or healpy map
        pivot spectral index with respect to which the moments are computed. If not specified, the pivot is taken to be the one cancelling first order (ensuring the quickest convergence for the expansion)
    tempbar: float or healpy map
        pivot temperature with respect to which the moments are computed. If not specified, the pivot is taken to be the one cancelling first order (ensuring the quickest convergence for the expansion)
     pivot_type: string
         either '2D' or '3D'. If '2D' the pivots are single floats for the whole sky. If '3D' the pivots are maps.

    Returns
    -------
    mom: array of size (2,border,torder,npix)
        numpy array containing all the moment maps in polarisation and intensity.
    betabari, betabar, b_Tbari, b_Tbar: flots or healpy maps
        values of the pivot spectral parameters of the expansion.
    """
    npix = hp.nside2npix(nside)
    mom = np.zeros([2,maxborder+1,maxtorder+1,npix],dtype='complex128')

    alisti = modelnu0[:,0]
    alist = modelnu0[:,1] + modelnu0[:,2]*1j
    betalist=betamap
    
    if betabar is None:
        if pivot_type=='2D':
            betabar = np.real(np.sum(betamap * alist) / np.sum(alist))
            betabari = np.sum(betamap * alisti) / np.sum(alisti)
        elif pivot_type=='3D':
            betabar = np.real(np.sum(betamap * alist, axis=0) / np.sum(alist, axis=0))
            betabari = np.sum(betamap * alisti, axis=0) / np.sum(alisti, axis=0)
    else:
        betabari=betabar

    print('pivot spectral index of expansion =%s'%betabar)

    if SED_type=="mbb":
        b_Tlist=1/tempmap
        if tempbar is None:
            if pivot_type=='2D':
                b_Tbar = np.real(np.sum(b_Tlist * alist) / np.sum(alist))
                b_Tbari = np.sum(b_Tlist * alisti) / np.sum(alisti)
            elif pivot_type=='3D':
                b_Tbar = np.real(np.sum(b_Tlist * alist, axis=0) / np.sum(alist, axis=0))
                b_Tbari = np.sum(b_Tlist * alisti, axis=0) / np.sum(alisti, axis=0)
        else:
            b_Tbar=1/tempbar
            b_Tbari=b_Tbar
        print('pivot temperature of expansion =%s'%(1/b_Tbar))
    elif SED_type=='pl':
        b_Tbar= None
        b_Tbari= None


    if pivot_type=='3D':
        for ipix in range(npix):
            if betabar[ipix]<betarange[0] or betabar[ipix]>betarange[1]:
                betabar[ipix]=betabari[ipix]
            if b_Tbar[ipix]<b_Trange[0] or b_Tbar[ipix]>b_Trange[1]:
                b_Tbar[ipix]=b_Tbari[ipix]

    if SED_type=='mbb':
        if maxtorder>=maxborder:
            for border in range(maxborder+1):
                    for torder in range(maxtorder+1-border):
                        if (border == 0)  * (torder == 0) == 1:
                            mom[0,0,0] = np.ones(npix)  
                            mom[1,0,0] = np.ones(npix) 
                        else:
                            mom[0,border,torder] = np.sum(alisti*(betalist-betabari)**border*(b_Tlist-b_Tbari)**torder,axis=0)/np.sum(alisti,axis=0)
                            mom[1,border,torder] = np.sum(alist*(betalist-betabar)**border*(b_Tlist-b_Tbar)**torder,axis=0)/np.sum(alist,axis=0)
        else:
            for torder in range(maxtorder+1):
                for border in range(maxborder+1-torder):
                    if (border == 0)  * (torder == 0) == 1:
                        mom[0,0,0] = np.ones(npix)   
                        mom[1,0,0] = np.ones(npix)   
                    else:
                        mom[0,border,torder] = np.sum(alisti*(betalist-betabari)**border*(b_Tlist-b_Tbari)**torder,axis=0)/np.sum(alisti,axis=0)
                        mom[1,border,torder] = np.sum(alist*(betalist-betabar)**border*(b_Tlist-b_Tbar)**torder,axis=0)/np.sum(alist,axis=0)
    
    elif SED_type=='pl':
        for border in range(maxborder+1):
            if border == 0:
                mom[0,0,0] = np.ones(npix)  
                mom[1,0,0] = np.ones(npix) 
            else:
                mom[0,border,0] = np.sum(alisti*(betalist-betabari)**border,axis=0)/np.sum(alisti,axis=0)
                mom[1,border,0] = np.sum(alist*(betalist-betabar)**border,axis=0)/np.sum(alist,axis=0)

    if pivot_type=='3D':
        mom[1,1,0] = 1j * mom[1,1,0].imag
        mom[1,0,1] = 1j * mom[1,0,1].imag 


    return mom, betabari, betabar, b_Tbari, b_Tbar
    
def model_SED_moments(nside,nu,model,mom,tempmap,nu0=353.,maxborder=3,maxtorder=3,nside_moments=512,mult_factor=1.,mom3D=None,mult_factor_3D=1.,tempmapP=None,SED_type='mbb'):
    """
    Compute the moment value in each pixel

    Parameters
    ----------
    nside: int
        Resolution parameter at which this model is to be calculated.
    modelnu0: triplet of healpy map of size npix
        Template of (I,Q,U) for the model at reference frequency nu0
    betamap: numpy array of size (nlayer,npix)
        Map of the beta values contained in each pixel. For two dimensional models (e.g. d1, d10...), must be of size (1,npix).
    tempmap: numpy array of size (nlayer,npix)
        Map of the temperature values contained in each pixel. For two dimensional models (e.g. d1, d10...), must be of size (1,npix).
    maxborder: float
        Maximum order of the expansion in beta at which the moments must be computed
    maxtorder: float
        Maximum order of the expansion in temp at which the moments must be computed
    SED_type: string
        Type of SED, either 'mbb' or 'pl'
    betabar: float or healpy map
        pivot spectral index with respect to which the moments are computed. If not specified, the pivot is taken to be the one cancelling first order (ensuring the quickest convergence for the expansion)
    tempbar: float or healpy map
        pivot temperature with respect to which the moments are computed. If not specified, the pivot is taken to be the one cancelling first order (ensuring the quickest convergence for the expansion)
     pivot_type: string
         either '2D' or '3D'. If '2D' the pivots are single floats for the whole sky. If '3D' the pivots are maps.

    Returns
    -------
    mom: array of size (2,border,torder,npix)
        numpy array containing all the moment maps in polarisation and intensity.
    betabari, betabar, b_Tbari, b_Tbar: flots or healpy maps
        values of the pivot spectral parameters of the expansion.
    """
    
    npix = hp.nside2npix(nside)
    map3D = np.zeros([3,npix])
    beta = sym.Symbol('ß')
    b_T = sym.Symbol('ß_T')
    
    if SED_type=='mbb':
        nuval = nu * 1e9
        nu0val = nu0 * 1e9
        Bval = 2*constants.h*(nuval**3)/constants.c**2
        Cval = constants.h*nuval/constants.k
        Bval0 = 2*constants.h*(nu0val**3)/constants.c**2
        Cval0 = constants.h*nu0val/constants.k
        Bvalratio = Bval/Bval0
        SED = ((nuval / nu0val) ** beta) * Bvalratio / (sym.exp(Cval*b_T) - 1) * (sym.exp(Cval0*b_T) - 1)

    elif SED_type =='pl':
        SED = (nu / nu0) ** beta * (b_T)**0 
    
    if maxtorder>=maxborder:
        for border in range(maxborder+1):
            for torder in range(maxtorder+1-border):
                analyticalmom = sym.diff(SED,beta,border)*sym.diff(SED,b_T,torder).factor()/SED**2
            
                if torder == 0:
                    valuemomi = float(analyticalmom)
                    valuemomP = valuemomi
                else:
                    if tempmapP is None:
                        analyticalmom = sym.lambdify(b_T,analyticalmom,'numpy')
                        valuemomi = analyticalmom(1/tempmap)
                        valuemomP = valuemomi
                    else:
                        analyticalmom = sym.lambdify(b_T,analyticalmom,'numpy')
                        valuemomi = analyticalmom(1/tempmap)
                        valuemomP = analyticalmom(1/tempmapP)
                if ((border == 0)  * (torder == 0)) == 1:
                    modelcomplex = (model[1]+1j*model[2]) * 1./(np.math.factorial(border)*np.math.factorial(torder))*mom[1,border,torder]*valuemomP
                    map3D[0] += model[0] * 1./(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom[0,border,torder])*valuemomi
                else:
                    modelcomplex = (model[1]+1j*model[2]) * mult_factor/(np.math.factorial(border)*np.math.factorial(torder))*mom[1,border,torder]*valuemomP
                    map3D[0] += model[0] * mult_factor/(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom[0,border,torder])*valuemomi
                    if mom3D is not None:
                        modelcomplex += (model[1]+1j*model[2]) * mult_factor_3D/(np.math.factorial(border)*np.math.factorial(torder))*mom3D[1,border,torder]*valuemomP
                        map3D[0] += model[0] * mult_factor_3D/(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom3D[0,border,torder])*valuemomi
                map3D[1] += np.real(modelcomplex)
                map3D[2] += np.imag(modelcomplex)
    else:
        for torder in range(maxtorder+1):
            for border in range(maxborder+1-torder):
                analyticalmom = sym.diff(SED,beta,border)*sym.diff(SED,b_T,torder).factor()/SED**2
            
                if torder == 0:
                    valuemomi = float(analyticalmom)
                    valuemomP = valuemomi
                else:
                    if tempmapP is None :
                        analyticalmom = sym.lambdify(b_T,analyticalmom,'numpy')
                        valuemomi = analyticalmom(1/tempmap)
                        valuemomP = valuemomi
                    else:
                        analyticalmom = sym.lambdify(b_T,analyticalmom,'numpy')
                        valuemomi = analyticalmom(1/tempmap)
                        valuemomP = analyticalmom(1/tempmapP)
                if ((border == 0)  * (torder == 0)) == 1:
                    modelcomplex = (model[1]+1j*model[2]) * 1./(np.math.factorial(border)*np.math.factorial(torder))*mom[1,border,torder]*valuemomP
                    map3D[0] += model[0] * 1./(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom[0,border,torder])*valuemomi
                else:
                    modelcomplex = (model[1]+1j*model[2]) * mult_factor/(np.math.factorial(border)*np.math.factorial(torder))*mom[1,border,torder]*valuemomP
                    map3D[0] += model[0] * mult_factor/(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom[0,border,torder])*valuemomi
                    if mom3D is not None:
                        modelcomplex += (model[1]+1j*model[2]) * mult_factor_3D/(np.math.factorial(border)*np.math.factorial(torder))*mom3D[1,border,torder]*valuemomP
                        map3D[0] += model[0] * mult_factor_3D/(np.math.factorial(border)*np.math.factorial(torder))*np.real(mom3D[0,border,torder])*valuemomi

                map3D[1] += np.real(modelcomplex)
                map3D[2] += np.imag(modelcomplex)
    return map3D
