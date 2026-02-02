import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import copy
import warnings
from scipy.stats import pearsonr
from scipy.stats import t as t_dist
import setuptools


def estimate_variogram(D, data, M:int, qd:float):
    '''
    Estimate the empirical variogram from distance matrix between vertices,
    and data value at each vertex. Estiamtion performed in M bins, ranging
    from min_distance to qd * max_distance, where min_distance and max_distance
    are the min and max distance in the distance matrix.
    
    Inputs:
    D: ndarray (N,N)
     Distance matrix between all vertices
    data: ndarray (N,)
     Data value at each vertex.
    M: int
     number of bins to estimate variogram.
    qd:
     determine the maximum distance to evaluate variogram

    Returns:
    v : ndarray (M,)
        Estimated variogram values, i.e., semivariance.
    h : ndarray (M,)
        lag distances.

    This is similar to variogram estimation in BrainSMASH but determining the max distance evaluated in a different way
    '''
    Dmax = qd * np.max(D)
    Dmin = np.min(D[D > 0])
    
    # Upper triangle without diagonal
    triu_indices = np.triu_indices_from(D, k=1)
    dval = D[triu_indices]
    row = triu_indices[0]
    col = triu_indices[1]
    
    mask = dval <= Dmax #data pairs falling within the distance range of analysis
    dval = dval[mask]
    row = row[mask]
    col = col[mask]

    h = np.linspace(Dmin, Dmax, M) # linearly spaced lag distances
    delta = (Dmax - Dmin) / (M - 1) * 0.5 
    sigma = 6 * delta
    v = np.zeros(M)
    # variogram estimation using gaussian smoothing kernel, same as BrainSMASH
    for i in range(M):
        w = np.exp(-((2.68 * np.abs(h[i] - dval)) ** 2) / (2 * sigma ** 2))
        diff_sq = (data[row] - data[col]) ** 2
        v[i] = 0.5 * np.sum(w * diff_sq) / np.sum(w)

    return v, h


def fit_variogram(h,v,D,PrecomputedVariance=None, nugget:bool=True):
    '''
    Fit stable variogram model to empirical variogram
    Inputs:
     h: empirical lag distances, ndarray (M,)
     v: empirical semivariance at lag distances, ndarray (M,)
     D: distance matrix, ndarray (N,N)
     PrecomputedVariance: precomputed sill, when None, set to the maximum semivariance in v
     nugget: bool indicator for using/not using nugget

    Returns:
     c_para: fitted covariance matrix, ndarray (N,N)
     b: fitted stable model params, ndarray (4,): (sill, phi, exponent, nugget)
     f: variogram function f(h) gives semivariance at lag distance h
     fcov: covariance function, fcov(h) gives covariance at lag distance h
    '''
    if PrecomputedVariance is None:
        PrecomputedVariance = np.max(v)
    x0 = np.asarray([PrecomputedVariance, np.min(h), 1.]) # initial guess of stable variogram parameters
    lb = np.asarray([0., 0., 0.]) # lower bound of estimation
    ub = np.asarray([2*PrecomputedVariance, np.inf, 2.]) # upper bound of estimation, set ub of sill to 2*PrecomputedVariance for stable inference
    # fit variogram model
    if not nugget:
        b, _ = curve_fit(stable_variogram_no_nugget, h, v, p0=x0, bounds=(lb, ub))
        b = np.append(b, 0.)
    else:
        x0 = np.append(x0, 0.)
        lb = np.append(lb, 0.)
        ub = np.append(ub, 0.5 * PrecomputedVariance) # set ub for nugget for stable inference at extreme short-range autocorrelation
        b, _ = curve_fit(stable_variogram, h, v, p0=x0, bounds=(lb, ub))
    f = lambda h: stable_variogram(h, *b)
    fcov = lambda h: stable_covariance_func(h, b)
    c_para = fcov(D) # off-diagonal components of covariance matrix
    np.fill_diagonal(c_para, b[0] + b[3]) # diagonal set to sill + nugget
    
    return c_para, b, f, fcov


def stable_variogram_no_nugget(h, b1, b2, b3):
    '''
    stable variogram model without nugget, defined as: semivariance = sill * (1-exp(-(h/range)**shape))
    Inputs:
     h: lag distance to be evaluated
     b1: sill
     b2: range parameter
     b3: shape
    Outputs:
     senuvaruabce at lag distance h
    '''
    return b1 * (1 - np.exp(-(h / b2) ** b3))

def stable_variogram(h, b1, b2, b3, b4):
    '''
    stable variogram model with nugget, defined as: semivariance = sill * (1-exp(-(h/range)**shape))+ nugget
    Inputs:
     h: lag distance to be evaluated
     b1: sill
     b2: range parameter
     b3: shape
    Outputs:
     senuvaruabce at lag distance h
    '''
    return b1 * (1 - np.exp(-(h / b2) ** b3)) + b4

def stable_covariance_func(h, b):
    '''
    covariance function based on stable variogram model for observations with distance h
    Equivalent to (sill + nugget) - (sill * (1-exp(-(h/range)**shape))+ nugget) = sill * exp(-(h/range)**shape) for h>0
    When h==0, set to sill+nugget, which is not computed here
    Inputs:
     h: lag distance at which to compute covariance
     b: parameters for stable models
    Outputs:
     Covariance at distance h
    '''
    b1, b2, b3 = b[:3]
    return (h > 0) * (b1 * np.exp(-(h / b2) ** b3))

def parc_data(parc, c_para, b, D, coord, max_clusters, min_clusters, min_cluster_size, map_idx):
    '''
    parcellate data depending on setting to account for nonstationarity
    3 scenarios:
     1. parc is None: do not parcellate and return covariance matrix c_para as is (new variable name fc_para)
     2. parc is string 'auto': determine the number of parcels based on estiamted range and shape parameter from stable variogram model (i.e., b[1] and b[2]), and parcellate data using spatial clustering
     3. parc is user specified np int array with shape (M,) with each int indicating a unique parcel: return parc as is (new variable name parc_out), raise warning if risk of over-parcellation (compared to 'auto')
    Inputs:
     parc: either None, 'auto', or (N,), specifying the setting of parcellation
     c_para: covariance matrix estimated from stationary ESSE, i.e., without parcel
     b: stable variogram model parameters estimated from stationary ESSE
     D: distance matrix of data (N, N)
     coord: (N, 3) spatial coordinates of data or None. If None, spatial clustering is conducted based on the distance matrix D with KMedoids
     max_clusters: maximum number of parcellation, set to avoid over-parcellation at weak autocorrelation, e.g., spatial independence
     min_clusters: minimum number of parcellation, set to 1 will allow nonstatioanry ESSE to collapse to stationary ESSE. This was used to mandate parcellation and test difference between sESSE and nESSE in the manuscript.
     min_cluster_size: set to avoid too small parcellations, we set to 500 in fsaverage5 mesh with 10k vertices
     map_idx: used to identify the map evaluated when raise warning
    Outputs:
     parc_out: None when there is no subdivision of parcels, or (N,) of int where each unique int indicate a parcel
     n_parc: number of parcels
     unique_parcs: index of unique parcels
     fc_para: covariance matrix, either as c_para (when no parcellation) or zeros (parcellated)
    '''
    if parc is None: # if None, return covariance matrix as is
        fc_para = c_para
        n_parc = 1
        unique_parcs = None
        parc_out = None
    else:
        # if not None, first compute number of parcels in data-driven manner depending on the strength of autocorrelation (i.e., the effective range of variogram)
        range_len = b[1] * 2.996 ** (1/b[2]) # effective range
        nPoints = np.max([np.sum(D < range_len) / D.shape[0] - 1, min_cluster_size]) # number of points per parcel on average, when parcel radius ~ effectuve rabge
        n_clusters = np.max([np.min([np.floor(D.shape[0] / nPoints), max_clusters]),min_clusters]).astype(int) # number of parcels
        if parc == 'auto':
            if coord is not None: # spatial clustering via Kmeans on coordinates
                parc_out = KMeans(n_clusters).fit(coord).labels_ # if coord available use kmeans
            else: # spatial clustering via KMedoids on distance matrix
                parc_out = KMedoids(n_clusters).fit(D).labels_ # if coord not available use kmedoids
            unique_parcs = np.unique(parc_out)
            n_parc = len(unique_parcs)
        else: # if user specified parcel, raise waring if risk of over-parcellation (more than estimated by 'auto')
            parc_out = parc # parcellation returned as is
            unique_parcs = np.unique(parc_out)
            n_parc = len(unique_parcs)
            if n_parc > n_clusters:
                warnings.warn(f'data No.{map_idx}: specified number of parcs {n_parc} is larger than data-derived max number of parcs {n_clusters}, carefully trade off the ability for detecting nonstationarity and the parcel coverage for robust estimation.')
        if n_parc == 1: # if does not subdivide, return covariance matrix as is
            fc_para = c_para
        else: # if subdivide, initialize and return a covariance matrix of zeros that will be filled in later steps, i.e., nESSE
            fc_para = np.zeros_like(c_para)
    return parc_out, n_parc, unique_parcs, fc_para


def effective_sample_size_estimation(x, y, coord=None, D=None, dim=None, M=None, qd=0.7, xparc=None, yparc=None, max_clusters=10, min_cluster_size=500, min_clusters=1, M_cluster=None, nugget=True):
    '''
    Main function that runs sESSE and nESSE to compute effective sample size and autocorrelation-corrected p-values
    Leave xparc=None and yparc=None will run sESSE, while setting to 'auto' or user-specified parcellation np int array (N,) will run nESSE
    Inputs:
     x, y: (N,) array of spatial map data to evaluate association. Can contain missing values such as NaN and Inf
     coord: (N, 3) array of spatial coordinates for observations, when unknown and leave to None, the function requires D to run sESSE, and D and dim to run nESSE
     D: (N, N) distance matrix. When leave to None, comnputed from coord
     dim: spatial dimension of data, when leave to None, computed as coord.shape[1] if needed
     M: number of lag distances to evaluate when estimate variogram - important hyperparameter that determines the quality of variogram estimation, large values preferred. When set to None, use 3*sqrt(N) as default
     qd: float (0,1], determine the covarage of lag distances evalauted in variogram, with maximum distance evaluated being qd*np.max(D) — important hyperparameter that determiens the quality of variogram estimation, large values preferred. Default 0.7
     xparc: None, 'auto', or (N,) int np.array that specified parcellation settings for map x, if np.array, index should begin from 0, i.e., 0 to Np - 1 if Np parcels specified.
     yparc: None, 'auto', or (N,) int np.array that specified parcellation settings for map y, if np.array, index should begin from 0, i.e., 0 to Np - 1 if Np parcels specified.
     max_clusters: maximum number of parcellatiosn allowed in nESSE
     min_clusters: minimum number of parcellations allowed in nESSE
     min_cluster_size: minimum size of parcellations (# observations per parcel)
     M_cluster: the number of lag distances to eavaluate in nESSE parcels when estimating their variograms. When set to None, default to 3*sqrt(Np), where Np is the number of observations in each parcel
     nugget: bool indicator of whether use nugget in variogram models or not. Default to True because nugget helps with discontineity at short distances, and set to false can result in problems especially when data are nonstationary
    Oututs:
     pef: significance p-values based on ESSE
     rX: Pearson correlation coefficient between x and y
     nef: effective sample size estimated
     run_status: 1 indicates successful run, and 0 indicates unsuccessful run such as when nef < 2 and data are too smooth to infer significance
     n_parc: [xn_parc, yn_parc] that indicates the number of parcels for each map in nESSE
     p_naive: Significance with independence assumption and without controlling for autocorrelation
     fc_para1: (Nvx, Nvx) covariance matrix for map x, with Nvx indicating the number of valid (i.e., finite value) observations in map x.
     fc_para2: (Nvy, Nvy) covariance matrix for map y, with Nvy indicating the number of valid (i.e., finite value) observations in map y.
    '''
    assert (coord is not None or D is not None), 'at least one of coord and D is required'
    assert ((coord is not None or dim is not None) or xparc is None and yparc is None), 'dim is required for nonstationary ESSE when coord is not provided'
    valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[valid]
    y = y[valid]
    x = stats.zscore(x)
    y = stats.zscore(y)
    if D is not None:
        D = D[np.ix_(valid, valid)]
    else:
        coord = coord[valid,:]
        D = squareform(pdist(coord))
        dim = coord.shape[1]

    if M is None:
        M = 3 * np.ceil(np.sqrt(x.shape[0])).astype('int')

    PrecomputedVariance = None
    v1,h1 = estimate_variogram(D, x, M, qd)
    v2,h2 = estimate_variogram(D, y, M, qd)
    c_para1, b1, f1, fcov1 = fit_variogram(h1,v1,D,PrecomputedVariance,nugget)
    c_para2, b2, f2, fcov2 = fit_variogram(h2,v2,D,PrecomputedVariance,nugget)

    xparc, xn_parc, xunique_parcs, fc_para1 = parc_data(xparc, c_para1, b1, D, coord, max_clusters, min_clusters, min_cluster_size, 1)
    yparc, yn_parc, yunique_parcs, fc_para2 = parc_data(yparc, c_para2, b2, D, coord, max_clusters, min_clusters, min_cluster_size, 1)
    if xn_parc > 1:
        exponent1 = b1[2]
        fc_para1, pb1 = fit_covariance_blocks(x, D, xn_parc, xparc, M_cluster, qd, nugget, exponent1)
        fc_para1 = process_convolution_crossblocks(fc_para1, pb1, x, D, xn_parc, xparc, dim, exponent1)
    if yn_parc > 1:
        exponent2 = b2[2]
        fc_para2, pb2 = fit_covariance_blocks(y, D, yn_parc, yparc, M_cluster, qd, nugget, exponent2)
        fc_para2 = process_convolution_crossblocks(fc_para2, pb2, y, D, yn_parc, yparc, dim, exponent2)

    nef = cov2nef(fc_para1,fc_para2)
    run_status = nef > 2

    rX, p_naive = pearsonr(x, y)
    if run_status:
        pef = nef2p(rX, nef)
    else:
        pef = np.nan
    n_parc = np.asarray([xn_parc, yn_parc])
    return pef, rX, nef, run_status, n_parc, p_naive, fc_para1, fc_para2

def covariance_estimation(x, coord=None, D=None, dim=None, M=None, qd=0.7, xparc=None, max_clusters=10, min_cluster_size=500, min_clusters=1, M_cluster=None, nugget=True):
    '''
    Comupute the covariance matrix for a single map x using sESSE or nESSE
    This can be particularly useful when pairwise association between a large umber of maps needs to be evaluated
    Compute the covariance matrix for each data separately and save for latter use can avoid repeatative covariance estimation in the effective_sample_size_estimation function
    Statistical significance between two maps can be inferred by loading saved covariance matrices (cov1 and cov2) of two maps (x and y), and compute effective sample size and p-values following steps below:
    get submatrix of covariance matrices for points that are valid in both maps - valid = np.isfinite(x) & np.isfinite(y), cov1 = cov1[np.ix_(valid, valid)], cov2 = cov2[np.ix_(valid, valid)], x = x[valid], y = y[valid]
    compute nef - cov2nef(cov1, cov2)
    compute test statistics such as Pearson correlation coefficient — rX, p_naive = pearsonr(x, y)
    compute significance p-value from test statistics and effective sample size: nef2p(rX, nef)
    Inputs are same as in effective_sampe_size_estimation but with y and yparc removed.
    Outputs: covmat - covariance matrix of map x in shape (N, N), where rows and columns corresponding to invalid observations in x (e.g., NaN, Inf) are set to np.nan and need to be removed before computing nef
    '''
    assert (coord is not None or D is not None), 'at least one of coord and D is required'
    assert ((coord is not None or dim is not None) or xparc is None), 'dim is required for nonstationary ESSE when coord is not provided'
    nx = len(x)
    valid = np.isfinite(x)
    x = x[valid]
    x = stats.zscore(x)
    covmat = np.full((nx, nx), np.nan)
    if D is not None:
        D = D[np.ix_(valid, valid)]
    else:
        coord = coord[valid,:]
        D = squareform(pdist(coord))
        dim = coord.shape[1]

    if M is None:
        M = 3 * np.ceil(np.sqrt(x.shape[0])).astype('int')

    PrecomputedVariance = None
    v1,h1 = estimate_variogram(D, x, M, qd)
    c_para1, b1, f1, fcov1 = fit_variogram(h1,v1,D,PrecomputedVariance,nugget)

    xparc, xn_parc, xunique_parcs, fc_para1 = parc_data(xparc, c_para1, b1, D, coord, max_clusters, min_clusters, min_cluster_size, 1)
    if xn_parc > 1:
        exponent1 = b1[2]
        fc_para1, pb1 = fit_covariance_blocks(x, D, xn_parc, xparc, M_cluster, qd, nugget, exponent1)
        fc_para1 = process_convolution_crossblocks(fc_para1, pb1, x, D, xn_parc, xparc, dim, exponent1)
    
    covmat[np.ix_(valid,valid)] = fc_para1
    return covmat


def cov2nef(c_para1, c_para2):
    '''
    Compute effective sample size from covariance matrices c_para1 and c_para2
    Below is a computational efficient implementation equivalent to nef=real(1/(trace(B*fc_para1*B*fc_para2)/(trace(B*fc_para1)*trace(B*fc_para2)))+1);
    '''
    c1 = c_para1 - np.mean(c_para1, axis=0, keepdims=True) - np.mean(c_para1, axis=1, keepdims=True) + np.mean(c_para1)
    c2 = c_para2 - np.mean(c_para2, axis=0, keepdims=True) - np.mean(c_para2, axis=1, keepdims=True) + np.mean(c_para2)
    num = np.trace(c1 @ c2)
    den = np.trace(c1) * np.trace(c2)
    nef = np.real(1 / (num / den) + 1)
    return nef

def nef2p(rX, nef):
    '''
    infer statistical significance p-value from test statistics rX and effective sample size nef
    '''
    df = max(0, nef - 2)
    if df == 0:
        return np.nan
    t = rX * np.sqrt(df / (1 - rX**2))
    p = 2 * t_dist.sf(np.abs(t), df)
    return p

def fit_covariance_blocks(x, D, n_clusters, point_cluster_idx, M_cluster, qd, nugget, exponent):
    '''
    fit variogram model for each parcel and compute the diagonal blocks of nonstationary covariance matrix
    Inputs:
     x: (N,) array of spatial map data to evaluate association. All values are valid.
     D: (N, N) distance matrix. 
     n_clusters: the number of parcels
     point_cluster_idx: (N,) int np.array that specified parcellation settings for map x, ranging from 0 to NP-1 if NP parcels.
     M_cluster: the number of lag distances to eavaluate in parcel when estimating their variograms. When set to None, default to 3*sqrt(Np), where Np is the number of observations in each parcel
     qd: float (0,1]
     nugget: bool indicator of whether use nugget in variogram models or not.
     exponent: the shape parameter estimated using global stationary variogram model. This will be kept the same across parcels to obtain valid nonstationary covariance expression (i.e., PSD matrix)
    Oututs:
     c_para: (N, N) covariance matrix for map x, where within parcel covariance are estimated but cross-parcel elements are set to 0
     b: (n_clusters, 4) array of stable variogram model parameters, each row corresponds a parcel.
    '''
    c_para = np.zeros(D.shape) # initiation
    b = np.zeros(shape=(n_clusters,4))
    computeM = (M_cluster is None)
    for i in np.arange(n_clusters):
        v_select = point_cluster_idx == i
        x_select = x[v_select]
        var_x_select = x_select.var()
        x_select = stats.zscore(x_select)
        D_select = D[np.ix_(v_select, v_select)]
        if computeM:
            M_cluster = 3 * np.ceil(np.sqrt(x_select.shape[0])).astype(int)
        v, h = estimate_variogram(D_select, x_select, M_cluster, qd)
        pc_para, pb, f, fcov = fit_variogram_fixed_exponent(h, v, D_select, exponent, 1, nugget)
        c_para[np.ix_(v_select, v_select)] = pc_para * var_x_select
        pb[0] = pb[0] * var_x_select
        pb[-1] = pb[-1] * var_x_select
        b[i,:] = pb
    return c_para, b

def fit_variogram_fixed_exponent(h, v, D, exponent, PrecomputedVariance=None, nugget: bool = True):
    '''
    same as fit_variogram, but for stable model with predetermined range parameter.
    '''
    if PrecomputedVariance is None:
        PrecomputedVariance = np.max(v)
    x0 = np.asarray([PrecomputedVariance, np.min(h)])
    lb = np.asarray([0., 0.])
    ub = np.asarray([2*PrecomputedVariance, np.inf])
    if not nugget:
        b, _ = curve_fit(lambda h, b1, b2: stable_variogram_fixed_exp_no_nugget(h, b1, b2, exponent), h, v, p0=x0, bounds=(lb, ub))
        b = np.array([b[0], b[1], exponent, 0.0])
    else:
        x0 = np.append(x0, 0.)
        lb = np.append(lb, 0.)
        ub = np.append(ub, 0.5*PrecomputedVariance) # set ub for nugget to avoid inaccurate shape parameter fitting when no/very-short-range autocorrelation
        b, _ = curve_fit(lambda h, b1, b2, b3: stable_variogram_fixed_exp(h, b1, b2, b3, exponent), h, v, p0=x0, bounds=(lb, ub))
        b = np.asarray([b[0], b[1], exponent, b[-1]])

    f = lambda h: stable_variogram(h, *b)
    fcov = lambda h: stable_covariance_func(h, b)
    c_para = fcov(D)
    np.fill_diagonal(c_para, b[0] + b[3])
    
    return c_para, b, f, fcov

def stable_variogram_fixed_exp_no_nugget(h, b1, b2, fixed_exp):
    '''
    stable variogram with prespecified shape parameter, without nugget (i.e., nugget set to 0)
    '''
    return b1 * (1 - np.exp(-(h / b2) ** fixed_exp))

def stable_variogram_fixed_exp(h, b1, b2, b3, fixed_exp):
    '''
    stable variogram with prespecified shape parameter, with nugget
    '''
    return b1 * (1 - np.exp(-(h / b2) ** fixed_exp)) + b3

def process_convolution_crossblocks(c_para, b, x, D, n_clusters, point_cluster_idx, dim, exponent):
    '''
    process convolution to infer the cross-parcel covariance of nonstationary covariance matrix
    Inputs:
     c_para: (N, N) the covariance matrix output from fit_covariance_blocks, where covariances are estimated for within parcel pairs but not cross-parcel
     b: (n_clusters, 4) array of fitted stable variogram model parameter, each row per parcel
     x: (N,) map data
     D: (N, N) distance matrix
     n_clusters: number of parcels
     point_cluster_idx: (N, ) of int array starting from 0 indicating the membership of each point to parcels
     dim: spatial dimension of the data
     exponent: shape parameter fitted using the global stationary variogram, kept the same for valid PSD covariance matrix
    Outputs:
     c_para: nonstationary covariance matrix by process convolution
    '''
    for i in np.arange(n_clusters-1):
        v_select1 = point_cluster_idx == i
        phi_i = b[i,1]
        for j in np.arange(i+1, n_clusters):
            v_select2 = point_cluster_idx == j
            phi_j = b[j,1]
            D_select = D[np.ix_(v_select1, v_select2)]
            sig = (phi_i ** 2 + phi_j ** 2) / 2
            Qij = D_select ** 2 / sig
            c_para[np.ix_(v_select1, v_select2)] = (phi_i * phi_j / sig) ** (dim/2) * np.sqrt(b[i,0] * b[j,0]) * np.exp(- np.sqrt(Qij) ** exponent)
            c_para[np.ix_(v_select2, v_select1)] = c_para[np.ix_(v_select1, v_select2)].T
    return c_para