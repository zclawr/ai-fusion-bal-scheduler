import os
import numpy as np
import xarray as xr
from omfit_classes.omfit_gacode import OMFITcgyro, OMFITgyro, OMFITinputgacode

# -*-Python-*-
# Created by neisert at 23 Oct 2023  16:59

"""
Using fraction of db[n:m], this script computes TGLF and QLGYRO fluxes using the same settings like azf+/-1
"""

n = 0
m = None  # 10 for debugging or None


# from classes.omfit_tglf import intensity_sat, sum_ky_spectrum, get_sat_params, get_zonal_mixing, linear_interpolation, mode_transition_function, flux_integrals, flux_integrals

def get_zonal_mixing(
    ky_mix,
    gamma_mix,
    **kw,
):
    """
    :param ky_mix: poloidal wavenumber [nk]
    :param gamma_mix: most unstable growth rates [nk]
    :param **kw: keyword list in input.tglf
    """
    nky = len(ky_mix)
    gammamax1 = gamma_mix[0]
    kymax1 = ky_mix[0]
    testmax1 = gammamax1 / kymax1
    jmax1 = 0
    kymin = 0
    testmax = 0.0
    j1 = 0
    kycut = 0.8 / kw["rho_ion"]
    if kw["ALPHA_ZF"] < 0:
        kymin = 0.173 * np.sqrt(2.0) / kw["rho_ion"]
    if kw["SAT_RULE"] in [2, 3]:
        kycut = kw["grad_r0_out"] * kycut
        kymin = kw["grad_r0_out"] * kymin

    for j in range(0, nky - 1):
        if ky_mix[j] <= kycut and ky_mix[j + 1] >= kymin:
            j1 = j
            kymax1 = ky_mix[j]
            testmax1 = gamma_mix[j] / kymax1
            if testmax1 > testmax:
                testmax = testmax1
                jmax_mix = j
    if testmax == 0.0:
        jmax_mix = j1
    kymax1 = ky_mix[jmax_mix]
    gammamax1 = gamma_mix[jmax_mix]
    if kymax1 < kymin:
        kymax1 = kymin
        gammamax1 = gamma_mix[0] + (gamma_mix[1] - gamma_mix[0]) * (kymin - ky_mix[0]) / (ky_mix[1] - ky_mix[0])
    if jmax_mix > 0 and jmax_mix < j1:
        jmax1 = jmax_mix
        f0 = gamma_mix[jmax1 - 1] / ky_mix[jmax1 - 1]
        f1 = gamma_mix[jmax1] / ky_mix[jmax1]
        f2 = gamma_mix[jmax1 + 1] / ky_mix[jmax1 + 1]
        deltaky = ky_mix[jmax1 + 1] - ky_mix[jmax1 - 1]
        x1 = (ky_mix[jmax1] - ky_mix[jmax1 - 1]) / deltaky
        a = f0
        b = (f1 - f0 * (1 - x1 * x1) - f2 * x1 * x1) / (x1 - x1 * x1)
        c = f2 - f0 - b
        xmax = -b / (2.0 * c)
        if ky_mix[jmax1 - 1] < kymin:
            xmin = (kymin - ky_mix[jmax1 - 1]) / deltaky
        else:
            xmin = 0.0
        if xmax >= 1.0:
            kymax1 = ky_mix[jmax1 + 1]
            gammamax1 = f2 * kymax1
        elif xmax < xmin:
            if xmin > 0.0:
                kymax1 = kymin
                gammamax1 = (a + b * xmin + c * xmin * xmin) * kymin
            else:
                kymax1 = ky_mix[jmax1 - 1]
                gammamax1 = f0 * kymax1
        else:
            kymax1 = ky_mix[jmax1 - 1] + deltaky * xmax
            gammamax1 = (a + b * xmax + c * xmax * xmax) * kymax1
    vzf_mix = gammamax1 / kymax1
    kymax_mix = kymax1
    return vzf_mix, kymax_mix, jmax_mix



def get_sat_params(sat_rule_in, ky, gammas, mts=5.0, ms=128, small=0.00000001, **kw):
    """
    This function calculates the scalar saturation parameters and spectral shift needed
    for the TGLF saturation rules, dependent on changes to 'tglf_geometry.f90' by Gary Staebler

    :mts: the number of points in the s-grid (flux surface contour)
    :ms: number of points along the arclength
    :ds: the arc length differential on a flux surface
    :R(ms): the major radius on the s-grid
    :Z(ms): the vertical coordinate on the s-grid
    :Bp(ms): the poloidal magnetic field on the s-grid normalized to B_unit
    :**kw: input.tglf
    """
    drmajdx_loc = kw["DRMAJDX_LOC"]
    drmindx_loc = kw["DRMINDX_LOC"]
    kappa_loc = kw["KAPPA_LOC"]
    s_kappa_loc = kw["S_KAPPA_LOC"]
    rmin_loc = kw["RMIN_LOC"]
    rmaj_loc = kw["RMAJ_LOC"]
    zeta_loc = kw["ZETA_LOC"]
    q_s = kw["Q_LOC"]
    q_prime_s = kw["Q_PRIME_LOC"]
    p_prime_s = kw["P_PRIME_LOC"]
    delta_loc = kw["DELTA_LOC"]
    s_delta_loc = kw["S_DELTA_LOC"]
    s_zeta_loc = kw["S_ZETA_LOC"]
    alpha_e_in = kw["ALPHA_E"]
    vexb_shear = kw["VEXB_SHEAR"]
    sign_IT = kw["SIGN_IT"]
    units = kw["UNITS"]
    mass_2 = kw["MASS_2"]
    taus_2 = kw["TAUS_2"]
    zs_2 = kw["ZS_2"]

    zmaj_loc = 0.0
    dzmajdx_loc = 0.0
    norm_ave = 0.0
    SAT_geo1_out = 0.0
    SAT_geo2_out = 0.0
    dlp = 0.0

    R = np.zeros(ms + 1)
    Z = np.zeros(ms + 1)
    Bp = np.zeros(ms + 1)
    Bt = np.zeros(ms + 1)
    B = np.zeros(ms + 1)
    b_geo = np.zeros(ms + 1)
    qrat_geo = np.zeros(ms + 1)
    sin_u = np.zeros(ms + 1)
    s_p = np.zeros(ms + 1)
    r_curv = np.zeros(ms + 1)
    psi_x = np.zeros(ms + 1)
    costheta_geo = np.zeros(ms + 1)

    pi_2 = 2 * np.pi
    if rmin_loc < 0.00001:
        rmin_loc = 0.00001
    vs_2 = np.sqrt(taus_2 / mass_2)
    gamma_reference_kx0 = gammas[0, :]

    # Miller geo
    rmin_s = rmin_loc
    Rmaj_s = rmaj_loc

    # compute the arclength around the flux surface:
    # initial values define dtheta
    theta = 0.0
    x_delta = np.arcsin(delta_loc)
    arg_r = theta + x_delta * np.sin(theta)
    darg_r = 1.0 + x_delta * np.cos(theta)
    arg_z = theta + zeta_loc * np.sin(2.0 * theta)
    darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)
    r_t = -rmin_loc * np.sin(arg_r) * darg_r
    z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z
    l_t = np.sqrt(r_t**2 + z_t**2)

    # scale dtheta by l_t to keep mts points in each ds interval of size pi_2/ms
    dtheta = pi_2 / (mts * ms * l_t)
    l_t1 = l_t
    arclength = 0.0

    while theta < pi_2:
        theta = theta + dtheta
        if theta > pi_2:
            theta = theta - dtheta
            dtheta = pi_2 - theta
            theta = pi_2

        arg_r = theta + x_delta * np.sin(theta)
        darg_r = 1.0 + x_delta * np.cos(theta)  # d(arg_r)/dtheta
        r_t = -rmin_loc * np.sin(arg_r) * darg_r  # dR/dtheta
        arg_z = theta + zeta_loc * np.sin(2.0 * theta)
        darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)  # d(arg_z)/dtheta
        z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z  # dZ/dtheta
        l_t = np.sqrt(r_t**2 + z_t**2)  # dl/dtheta
        arclength = arclength + 0.50 * (l_t + l_t1) * dtheta  # arclength along flux surface in poloidal direction
        l_t1 = l_t

    # Find the theta points which map to an equally spaced s-grid of ms points along the arclength
    # going clockwise from the outboard midplane around the flux surface
    # by searching for the theta where dR**2 + dZ**2 >= ds**2 for a centered difference df=f(m+1)-f(m-1).
    # This keeps the finite difference error of dR/ds, dZ/ds on the s-grid small
    ds = arclength / ms
    t_s = np.zeros(ms + 1)
    t_s[ms] = -pi_2

    # Make a first guess based on theta = 0.0
    theta = 0.0
    arg_r = theta + x_delta * np.sin(theta)
    darg_r = 1.0 + x_delta * np.cos(theta)
    arg_z = theta + zeta_loc * np.sin(2.0 * theta)
    darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)
    r_t = -rmin_loc * np.sin(arg_r) * darg_r
    z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z
    l_t = np.sqrt(r_t**2 + z_t**2)
    dtheta = -ds / l_t
    theta = dtheta
    l_t1 = l_t

    for m in range(1, int(ms / 2) + 1):
        arg_r = theta + x_delta * np.sin(theta)
        darg_r = 1.0 + x_delta * np.cos(theta)
        arg_z = theta + zeta_loc * np.sin(2.0 * theta)
        darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)
        r_t = -rmin_loc * np.sin(arg_r) * darg_r
        z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z
        l_t = np.sqrt(r_t**2 + z_t**2)
        dtheta = -ds / (0.5 * (l_t + l_t1))
        t_s[m] = t_s[m - 1] + dtheta
        theta = t_s[m] + dtheta
        l_t1 = l_t

    # distribute endpoint error over interior points
    dtheta = (t_s[int(ms / 2)] - (-np.pi)) / (ms / 2)

    for m in range(1, int(ms / 2) + 1):
        t_s[m] = t_s[m] - (m) * dtheta
        t_s[ms - m] = -pi_2 - t_s[m]
    # Quinn additions,
    B_unit_out = np.zeros(ms + 1)
    grad_r_out = np.zeros(ms + 1)
    for m in range(0, ms + 1):
        theta = t_s[m]
        arg_r = theta + x_delta * np.sin(theta)
        darg_r = 1.0 + x_delta * np.cos(theta)
        arg_z = theta + zeta_loc * np.sin(2.0 * theta)
        darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)

        R[m] = rmaj_loc + rmin_loc * np.cos(arg_r)  # = R(theta)
        Z[m] = zmaj_loc + kappa_loc * rmin_loc * np.sin(arg_z)  # = Z(theta)

        R_t = -rmin_loc * np.sin(arg_r) * darg_r  # = dR/dtheta
        Z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z  # = dZ/dtheta

        l_t = np.sqrt(R_t**2 + Z_t**2)  # = dl/dtheta

        R_r = (
            drmajdx_loc + drmindx_loc * np.cos(arg_r) - np.sin(arg_r) * s_delta_loc * np.sin(theta) / np.sqrt(1.0 - delta_loc**2)
        )  # = dR/dr
        Z_r = (
            dzmajdx_loc
            + kappa_loc * np.sin(arg_z) * (drmindx_loc + s_kappa_loc)
            + kappa_loc * np.cos(arg_z) * s_zeta_loc * np.sin(2.0 * theta)
        )

        det = R_r * Z_t - R_t * Z_r  # Jacobian
        grad_r = abs(l_t / det)
        if m == 0:
            B_unit = 1.0 / grad_r  # B_unit choosen to make qrat_geo(0)/b_geo(0)=1.0
            if drmindx_loc == 1.0:
                B_unit = 1.0  # Waltz-Miller convention
        B_unit_out[m] = B_unit
        grad_r_out[m] = grad_r

        Bp[m] = (rmin_s / (q_s * R[m])) * grad_r * B_unit
        p_prime_s = p_prime_s * B_unit
        q_prime_s = q_prime_s / B_unit
        psi_x[m] = R[m] * Bp[m]

    delta_s = 12.0 * ds
    ds2 = 12.0 * ds**2
    for m in range(0, ms + 1):
        m1 = (ms + m - 2) % ms
        m2 = (ms + m - 1) % ms
        m3 = (m + 1) % ms
        m4 = (m + 2) % ms
        R_s = (R[m1] - 8.0 * R[m2] + 8.0 * R[m3] - R[m4]) / delta_s
        Z_s = (Z[m1] - 8.0 * Z[m2] + 8.0 * Z[m3] - Z[m4]) / delta_s
        s_p[m] = np.sqrt(R_s**2 + Z_s**2)
        R_ss = (-R[m1] + 16.0 * R[m2] - 30.0 * R[m] + 16.0 * R[m3] - R[m4]) / ds2
        Z_ss = (-Z[m1] + 16.0 * Z[m2] - 30.0 * Z[m] + 16.0 * Z[m3] - Z[m4]) / ds2
        r_curv[m] = (s_p[m] ** 3) / (R_s * Z_ss - Z_s * R_ss)
        sin_u[m] = -Z_s / s_p[m]

    # Compute f=R*Bt such that the eikonal S which solves
    # B*Grad(S)=0 has the correct quasi-periodicity S(s+Ls)=S(s)-2*pi*q_s, where Ls = arclength
    f = 0.0
    for m in range(1, ms + 1):
        f = f + 0.5 * ds * (s_p[m - 1] / (R[m - 1] * psi_x[m - 1]) + s_p[m] / (R[m] * psi_x[m]))
    f = pi_2 * q_s / f

    for m in range(0, ms + 1):
        Bt[m] = f / R[m]
        B[m] = np.sqrt(Bt[m] ** 2 + Bp[m] ** 2)
        qrat_geo[m] = (rmin_s / R[m]) * (B[m] / Bp[m]) / q_s
        b_geo[m] = B[m]
        costheta_geo[m] = -Rmaj_s * (Bp[m] / (B[m]) ** 2) * (Bp[m] / r_curv[m] - (f**2 / (Bp[m] * (R[m]) ** 3)) * sin_u[m])

    for m in range(1, ms + 1):
        dlp = s_p[m] * ds * (0.5 / Bp[m] + 0.5 / Bp[m - 1])
        norm_ave += dlp
        SAT_geo1_out += dlp * ((b_geo[0] / b_geo[m - 1]) ** 4 + (b_geo[0] / b_geo[m]) ** 4) / 2.0
        SAT_geo2_out += dlp * ((qrat_geo[0] / qrat_geo[m - 1]) ** 4 + (qrat_geo[0] / qrat_geo[m]) ** 4) / 2.0

    SAT_geo1_out = SAT_geo1_out / norm_ave
    SAT_geo2_out = SAT_geo2_out / norm_ave

    if units == "GYRO" and sat_rule_in == 1:
        SAT_geo1_out = 1.0
        SAT_geo2_out = 1.0

    R_unit = Rmaj_s * b_geo[0] / (qrat_geo[0] * costheta_geo[0])
    B_geo0_out = b_geo[0]
    Bt0_out = f / Rmaj_s
    grad_r0_out = b_geo[0] / qrat_geo[0]
    # Additional outputs for SAT2 G1(theta), Gq(theta)
    theta_out = t_s  # theta grid over which everything is calculated.
    Bt_out = B  # total magnetic field matching theta_out grid.

    # Compute spetral shift kx0_e
    vexb_shear_s = vexb_shear * sign_IT
    vexb_shear_kx0 = alpha_e_in * vexb_shear_s

    kx0_factor = abs(b_geo[0] / qrat_geo[0] ** 2)
    kx0_factor = 1.0 + 0.40 * (kx0_factor - 1.0) ** 2

    kyi = ky * vs_2 * mass_2 / abs(zs_2)
    wE = kx0_factor * np.array([min(x / 0.3, 1.0) for x in kyi]) * vexb_shear_kx0 / gamma_reference_kx0
    kx0_e = -(0.36 * vexb_shear_kx0 / gamma_reference_kx0 + 0.38 * wE * np.tanh((0.69 * wE) ** 6))

    if sat_rule_in == 1:
        if units == "CGYRO":
            wE = 0.0
            kx0_factor = 1.0
        kx0_e = -(0.53 * vexb_shear_kx0 / gamma_reference_kx0 + 0.25 * wE * np.tanh((0.69 * wE) ** 6))
    elif sat_rule_in == 2 or sat_rule_in == 3:
        kw["grad_r0_out"] = grad_r0_out
        kw["SAT_RULE"] = sat_rule_in
        if bool(kw["USE_AVE_ION_GRID"]):
            indices = [is_ for is_ in range(2, kw['NS'] + 1) if (kw[f'ZS_{is_}'] * kw[f'AS_{is_}']) / abs(kw['AS_1'] * kw['ZS_1']) > 0.1]

            charge = sum(kw[f'ZS_{is_}'] * kw[f'AS_{is_}'] for is_ in indices)
            rho_ion = sum(
                kw[f'ZS_{is_}'] * kw[f'AS_{is_}'] * (kw[f'MASS_{is_}'] * kw[f'TAUS_{is_}']) ** 0.5 / kw[f'ZS_{is_}'] for is_ in indices
            )

            rho_ion /= charge if charge != 0 else 1
        else:
            rho_ion = (kw['MASS_2'] * kw['TAUS_2']) ** 0.5 / kw['ZS_2']
        kw['rho_ion'] = rho_ion
        vzf_out, kymax_out, _ = get_zonal_mixing(ky, gamma_reference_kx0, **kw)
        if abs(kymax_out * vzf_out * vexb_shear_kx0) > small:
            kx0_e = -0.32 * ((ky / kymax_out) ** 0.3) * vexb_shear_kx0 / (ky * vzf_out)
        else:
            kx0_e = np.zeros(len(ky))
    a0 = 1.3
    if sat_rule_in == 1:
        a0 = 1.45
    elif sat_rule_in == 2 or sat_rule_in == 3:
        a0 = 1.6
    kx0_e = np.array([min(abs(x), a0) * x / abs(x) for x in kx0_e])
    kx0_e[np.isnan(kx0_e)] = 0

    return (
        kx0_e,
        SAT_geo1_out,
        SAT_geo2_out,
        R_unit,
        Bt0_out,
        B_geo0_out,
        grad_r0_out,
        theta_out,
        Bt_out,
        grad_r_out,
        B_unit_out,
    )


def mode_transition_function(x, y1, y2, x_ITG, x_TEM):
    if x < x_ITG:
        y = y1
    elif x > x_TEM:
        y = y2
    else:
        y = y1 * ((x_TEM - x) / (x_TEM - x_ITG)) + y2 * ((x - x_ITG) / (x_TEM - x_ITG))
    return y


def linear_interpolation(x, y, x0):
    i = 0
    while x[i] < x0:
        i += 1
    y0 = ((y[i] - y[i - 1]) * x0 + (x[i] * y[i - 1] - x[i - 1] * y[i])) / (x[i] - x[i - 1])
    return y0


def intensity_sat(
    sat_rule_in,
    ky_spect,
    gp,
    kx0_e,
    nmodes,
    QL_data,
    expsub=2.0,
    alpha_zf_in=1.0,
    kx_geo0_out=1.0,
    SAT_geo_out=1.0,
    bz1=0.0,
    bz2=0.0,
    return_phi_params=False,
    **kw,
):
    """
    TGLF SAT1 from [Staebler et al., 2016, PoP], SAT2 from [Staebler et al., NF, 2021] and [Staebler et al., PPCF, 2021],
    and SAT3 [Dudding et al., NF, 2022] takes both CGYRO and TGLF outputs as inputs

    :param sat_rule_in: saturation rule [1, 2, 3]

    :param ky_spect: poloidal wavenumber [nk]

    :param gp: growth rates [nk, nm]

    :param kx0_e: spectral shift of the radial wavenumber due to VEXB_SHEAR [nk]

    :param nmodes_in: number of modes stored in quasi-linear weights [1, ..., 5]

    :param QL_data: Quasi-linear weights [ky, nm, ns, nf, type (i.e. particle,energy,stress_tor,stress_para,exchange)]

    :param expsub: scalar exponent in gammaeff calculation [2.0]

    :param alpha_zf_in: scalar switch for the zonal flow coupling coefficient [1.0]

    :param kx_geo_out: scalar switch for geometry [1.0]

    :param SAT_geo_out: scalar switch for geoemtry [1.0]

    :param bz1: scalar correction to zonal flow mixing term [0.0]

    :param bz2: scalar correction to zonal flow mixing term [0.0]

    :param return_phi_params: bool, option to return parameters for calculing the SAT1, SAT2 model for phi [False]

    :param **kw: keyword list in input.tglf
    """
    if bool(kw["USE_AVE_ION_GRID"]):
        indices = [is_ for is_ in range(2, kw['NS'] + 1) if (kw[f'ZS_{is_}'] * kw[f'AS_{is_}']) / abs(kw['AS_1'] * kw['ZS_1']) > 0.1]

        charge = sum(kw[f'ZS_{is_}'] * kw[f'AS_{is_}'] for is_ in indices)
        rho_ion = sum(
            kw[f'ZS_{is_}'] * kw[f'AS_{is_}'] * (kw[f'MASS_{is_}'] * kw[f'TAUS_{is_}']) ** 0.5 / kw[f'ZS_{is_}'] for is_ in indices
        )

        rho_ion /= charge if charge != 0 else 1
    else:
        rho_ion = (kw['MASS_2'] * kw['TAUS_2']) ** 0.5 / kw['ZS_2']
    kw['rho_ion'] = rho_ion

    nky = len(ky_spect)
    if len(np.shape(gp)) > 1:
        gammas1 = gp[:, 0]  # SAT1 and SAT2 use the growth rates of the most unstable modes
    else:
        gammas1 = gp
    gamma_net = np.zeros(nky)

    if sat_rule_in == 1:
        etg_streamer = 1.05
        kyetg = etg_streamer / kw["rho_ion"]
        measure = np.sqrt(kw["TAUS_1"] * kw["MASS_2"])

    czf = abs(alpha_zf_in)
    small = 1.0e-10
    cz1 = 0.48 * czf
    cz2 = 1.0 * czf
    cky = 3.0
    sqcky = np.sqrt(cky)
    cnorm = 14.29

    if sat_rule_in in [2, 3]:
        kw["UNITS"] = "CGYRO"
        units_in = kw["UNITS"]
    else:
        units_in = kw["UNITS"]

    kycut = 0.8 / kw["rho_ion"]
    # ITG/ETG-scale separation (for TEM scales see [Creely et al., PPCF, 2019])

    vzf_out, kymax_out, jmax_out = get_zonal_mixing(ky_spect, gammas1, **kw)

    if kw["RLNP_CUTOFF"] > 0.0:
        ptot = 0
        dlnpdr = 0
        for i in range(1, kw["NS"] + 1, 1):
            ptot += kw["AS_%s" % i] * kw["TAUS_%s" % i]  # only kinetic species
            dlnpdr += kw["AS_%s" % i] * kw["TAUS_%s" % i] * (kw["RLNS_%s" % i] + kw["RLTS_%s" % i])
        dlnpdr = kw["RMAJ_LOC"] * dlnpdr / max(ptot, 0.01)

        if dlnpdr >= kw["RLNP_CUTOFF"]:
            dlnpdr = kw["RLNP_CUTOFF"]
        if dlnpdr < 4.0:
            dlnpdr = 4.0
    else:
        dlnpdr = 12.0

    if sat_rule_in == 2 or sat_rule_in == 3:
        # SAT2 fit for CGYRO linear modes NF 2021 paper
        b0 = 0.76
        b1 = 1.22
        b2 = 3.74
        if nmodes > 1:
            b2 = 3.55
        b3 = 1.0
        d1 = (kw["Bt0_out"] / kw["B_geo0_out"]) ** 4  # PPCF paper 2020
        d1 = d1 / kw["grad_r0_out"]
        # WARNING: this is correct, but it's the reciprocal in the paper (typo in paper)
        Gq = kw["B_geo0_out"] / kw["grad_r0_out"]
        d2 = b3 / Gq**2
        cnorm = b2 * (12.0 / dlnpdr)
        kyetg = 1000.0  # does not impact SAT2
        cky = 3.0
        sqcky = np.sqrt(cky)
        kycut = b0 * kymax_out
        cz1 = 0.0
        cz2 = 1.05 * czf
        measure = 1.0 / kymax_out

    if sat_rule_in == 3:
        kmax = kymax_out
        gmax = vzf_out * kymax_out
        kmin = 0.685 * kmax
        aoverb = -1.0 / (2 * kmin)
        coverb = -0.751 * kmax
        kT = 1.0 / kw["rho_ion"]  # SAT3 used up to ky rho_av = 1.0, then SAT2
        k0 = 0.6 * kmin
        kP = 2.0 * kmin
        c_1 = -2.42
        x_ITG = 0.8
        x_TEM = 1.0
        Y_ITG = 3.3 * (gmax**2) / (kmax**5)
        Y_TEM = 12.7 * (gmax**2) / (kmax**4)
        scal = 0.82  # Q(SAT3 GA D) / (2 * QLA(ITG,Q) * Q(SAT2 GA D))

        Ys = np.zeros(nmodes)
        xs = np.zeros(nmodes)

        for k in range(1, nmodes + 1):
            sum_W_i = 0

            # sum over ion species, requires electrons to be species 1
            for is_ in range(2, np.shape(QL_data)[2] + 1):
                sum_W_i += QL_data[:, k - 1, is_ - 1, 0, 1]

            # check for singularities in weight ratio near kmax
            i = 1
            while ky_spect[i - 1] < kmax:
                i += 1

            if sum_W_i[i - 1] == 0.0 or sum_W_i[i - 2] == 0.0:
                x = 0.5
            else:
                abs_W_ratio = np.abs(QL_data[:, k - 1, 0, 0, 1] / sum_W_i)
                abs_W_ratio = np.nan_to_num(abs_W_ratio)
                x = linear_interpolation(ky_spect, abs_W_ratio, kmax)

            xs[k - 1] = x
            Y = mode_transition_function(x, Y_ITG, Y_TEM, x_ITG, x_TEM)
            Ys[k - 1] = Y

    ax = 0.0
    ay = 0.0
    exp_ax = 1
    if kw["ALPHA_QUENCH"] == 0.0:
        if sat_rule_in == 1:
            # spectral shift model parameters
            ax = 1.15
            ay = 0.56
            exp_ax = 4
        elif sat_rule_in == 2 or sat_rule_in == 3:
            ax = 1.21
            ay = 1.0
            exp_ax = 2
            units_in = "CGYRO"

    for j in range(0, nky):
        kx = kx0_e[j]
        if sat_rule_in == 2 or sat_rule_in == 3:
            ky0 = ky_spect[j]
            if ky0 < kycut:
                kx_width = kycut / kw["grad_r0_out"]
            else:
                kx_width = kycut / kw["grad_r0_out"] + b1 * (ky0 - kycut) * Gq
            kx = kx * ky0 / kx_width
        gamma_net[j] = gammas1[j] / (1.0 + abs(ax * kx) ** exp_ax)

    if sat_rule_in == 1:
        vzf_out, kymax_out, jmax_out = get_zonal_mixing(ky_spect, gamma_net, **kw)
    else:
        vzf_out_fp = vzf_out
        vzf_out = vzf_out * gamma_net[jmax_out] / max(gammas1[jmax_out], small)

    gammamax1 = vzf_out * kymax_out
    kymax1 = kymax_out
    jmax1 = jmax_out
    vzf1 = vzf_out

    # include zonal flow effects on growth rate model:
    gamma_mix1 = np.zeros(nky)
    gamma = np.zeros(nky)

    for j in range(0, nky):
        gamma0 = gamma_net[j]
        ky0 = ky_spect[j]
        if sat_rule_in == 1:
            if ky0 < kymax1:
                gamma[j] = max(gamma0 - cz1 * (kymax1 - ky0) * vzf1, 0.0)
            else:
                gamma[j] = cz2 * gammamax1 + max(gamma0 - cz2 * vzf1 * ky0, 0.0)
        elif sat_rule_in == 2 or sat_rule_in == 3:
            if ky0 < kymax1:
                gamma[j] = gamma0
            else:
                gamma[j] = gammamax1 + max(gamma0 - cz2 * vzf1 * ky0, 0.0)
        gamma_mix1[j] = gamma[j]

    # Mix over ky>kymax with integration weight
    mixnorm1 = np.zeros(nky)
    for j in range(jmax1 + 2, nky):
        gamma_ave = 0.0
        mixnorm1 = ky_spect[j] * (
            np.arctan(sqcky * (ky_spect[nky - 1] / ky_spect[j] - 1.0)) - np.arctan(sqcky * (ky_spect[jmax1 + 1] / ky_spect[j] - 1.0))
        )
        for i in range(jmax1 + 1, nky - 1):
            ky_1 = ky_spect[i]
            ky_2 = ky_spect[i + 1]
            mix1 = ky_spect[j] * (np.arctan(sqcky * (ky_2 / ky_spect[j] - 1.0)) - np.arctan(sqcky * (ky_1 / ky_spect[j] - 1.0)))
            delta = (gamma[i + 1] - gamma[i]) / (ky_2 - ky_1)
            mix2 = ky_spect[j] * mix1 + (ky_spect[j] * ky_spect[j] / (2.0 * sqcky)) * (
                np.log(cky * (ky_2 - ky_spect[j]) ** 2 + ky_spect[j] ** 2) - np.log(cky * (ky_1 - ky_spect[j]) ** 2 + ky_spect[j] ** 2)
            )
            gamma_ave = gamma_ave + (gamma[i] - ky_1 * delta) * mix1 + delta * mix2
        gamma_mix1[j] = gamma_ave / mixnorm1

    if sat_rule_in == 3:
        gamma_fp = np.zeros_like(ky_spect)  # Assuming ky_spect is a numpy array
        gamma = np.zeros_like(ky_spect)  # Assuming ky_spect is a numpy array

        for j in range(1, nky + 1):
            gamma0 = gammas1[j - 1]
            ky0 = ky_spect[j - 1]

            if ky0 < kymax1:
                gamma[j - 1] = gamma0
            else:
                gamma[j - 1] = (gammamax1 * (vzf_out_fp / vzf_out)) + max(gamma0 - cz2 * vzf_out_fp * ky0, 0.0)

            gamma_fp[j - 1] = gamma[j - 1]

        # USE_MIX is true by default
        for j in range(jmax1 + 3, nky + 1):  # careful: I'm switching here to Fortran indexing, but found jmax1 using python indexing
            gamma_ave = 0.0
            ky0 = ky_spect[j - 1]
            kx = kx0_e[j - 1]

            mixnorm = ky0 * (np.arctan(sqcky * (ky_spect[nky - 1] / ky0 - 1.0)) - np.arctan(sqcky * (ky_spect[jmax1 + 1] / ky0 - 1.0)))

            for i in range(jmax1 + 2, nky):  # careful: I'm switching here to Fortran indexing, but found jmax1 using python indexing
                ky1 = ky_spect[i - 1]
                ky2 = ky_spect[i]
                mix1 = ky0 * (np.arctan(sqcky * (ky2 / ky0 - 1.0)) - np.arctan(sqcky * (ky1 / ky0 - 1.0)))
                delta = (gamma[i] - gamma[i - 1]) / (ky2 - ky1)
                mix2 = ky0 * mix1 + (ky0 * ky0 / (2.0 * sqcky)) * (
                    np.log(cky * (ky2 - ky0) ** 2 + ky0**2) - np.log(cky * (ky1 - ky0) ** 2 + ky0**2)
                )
                gamma_ave += (gamma[i - 1] - ky1 * delta) * mix1 + delta * mix2
            gamma_fp[j - 1] = gamma_ave / mixnorm

    if sat_rule_in == 3:
        if ky_spect[-1] >= kT:
            dummy_interp = np.zeros_like(ky_spect)
            k = 0
            while ky_spect[k] < kT:
                k += 1

            for i in range(k - 1, k + 1):
                gamma0 = gp[i, 0]
                ky0 = ky_spect[i]
                kx = kx0_e[i]

                if ky0 < kycut:
                    kx_width = kycut / kw["grad_r0_out"]
                    sat_geo_factor = kw["SAT_geo0_out"] * d1 * kw["SAT_geo1_out"]
                else:
                    kx_width = kycut / kw["grad_r0_out"] + b1 * (ky0 - kycut) * Gq
                    sat_geo_factor = kw["SAT_geo0_out"] * (d1 * kw["SAT_geo1_out"] * kycut + (ky0 - kycut) * d2 * kw["SAT_geo2_out"]) / ky0

                kx = kx * ky0 / kx_width
                gammaeff = 0.0
                if gamma0 > small:
                    gammaeff = gamma_fp[i]
                # potentials without multimode and ExB effects, added later
                dummy_interp[i] = scal * measure * cnorm * (gammaeff / (kx_width * ky0)) ** 2
                if units_in != "GYRO":
                    dummy_interp[i] = sat_geo_factor * dummy_interp[i]
            YT = linear_interpolation(ky_spect, dummy_interp, kT)
            YTs = np.array([YT] * nmodes)
        else:
            if aoverb * (kP**2) + kP + coverb - ((kP - kT) * (2 * aoverb * kP + 1)) == 0:
                YTs = np.zeros(nmodes)
            else:
                YTs = np.zeros(nmodes)
                for i in range(1, nmodes + 1):
                    YTs[i - 1] = Ys[i - 1] * (
                        ((aoverb * (k0**2) + k0 + coverb) / (aoverb * (kP**2) + kP + coverb - ((kP - kT) * (2 * aoverb * kP + 1))))
                        ** abs(c_1)
                    )

    # preallocate [nky] arrays for phi_params
    gammaeff_out = np.zeros((nky, nmodes))
    sig_ratio_out = np.zeros((nky, nmodes))  # SAT3
    kx_width_out = np.zeros(nky)
    sat_geo_factor_out = np.zeros(nky)
    # intensity
    field_spectrum_out = np.zeros((nky, nmodes))
    for j in range(0, nky):
        gamma0 = gp[j, 0]
        ky0 = ky_spect[j]
        kx = kx0_e[j]
        if sat_rule_in == 1:
            sat_geo_factor = kw["SAT_geo0_out"]
            kx_width = ky0
        if sat_rule_in == 2 or sat_rule_in == 3:
            if ky0 < kycut:
                kx_width = kycut / kw["grad_r0_out"]
                sat_geo_factor = kw["SAT_geo0_out"] * d1 * kw["SAT_geo1_out"]
            else:
                kx_width = kycut / kw["grad_r0_out"] + b1 * (ky0 - kycut) * Gq
                sat_geo_factor = kw["SAT_geo0_out"] * (d1 * kw["SAT_geo1_out"] * kycut + (ky0 - kycut) * d2 * kw["SAT_geo2_out"]) / ky0
            kx = kx * ky0 / kx_width

        if sat_rule_in == 1 or sat_rule_in == 2:
            for i in range(0, nmodes):
                gammaeff = 0.0
                if gamma0 > small:
                    gammaeff = gamma_mix1[j] * (gp[j, i] / gamma0) ** expsub
                if ky0 > kyetg:
                    gammaeff = gammaeff * np.sqrt(ky0 / kyetg)
                field_spectrum_out[j, i] = measure * cnorm * ((gammaeff / (kx_width * ky0)) / (1.0 + ay * kx**2)) ** 2
                if units_in != "GYRO":
                    field_spectrum_out[j, i] = sat_geo_factor * field_spectrum_out[j, i]
                # add these outputs
                gammaeff_out[j, i] = gammaeff
            kx_width_out[j] = kx_width
            sat_geo_factor_out[j] = sat_geo_factor

        elif sat_rule_in == 3:
            # First part
            if gamma_fp[j] == 0:
                Fky = 0.0
            else:
                Fky = (gamma_mix1[j] / gamma_fp[j]) ** 2 / (1.0 + ay * (kx**2)) ** 2
            for i in range(1, nmodes + 1):
                field_spectrum_out[j, i - 1] = 0.0
                gammaeff = 0.0
                if gamma0 > small:
                    if ky0 <= kP:  # initial quadratic
                        sig_ratio = (aoverb * (ky0**2) + ky0 + coverb) / (aoverb * (k0**2) + k0 + coverb)
                        field_spectrum_out[j, i - 1] = Ys[i - 1] * (sig_ratio**c_1) * Fky * (gp[j, i - 1] / gamma0) ** (2 * expsub)
                    elif ky0 <= kT:  # connecting quadratic
                        if YTs[i - 1] == 0.0 or kP == kT:
                            field_spectrum_out[j, i - 1] = 0.0
                        else:
                            doversig0 = ((Ys[i - 1] / YTs[i - 1]) ** (1.0 / abs(c_1))) - (
                                (aoverb * (kP**2) + kP + coverb - ((kP - kT) * (2 * aoverb * kP + 1)))
                                / (aoverb * (k0**2) + k0 + coverb)
                            )
                            doversig0 = doversig0 * (1.0 / ((kP - kT) ** 2))
                            eoversig0 = -2 * doversig0 * kP + ((2 * aoverb * kP + 1) / (aoverb * (k0**2) + k0 + coverb))
                            foversig0 = ((Ys[i - 1] / YTs[i - 1]) ** (1.0 / abs(c_1))) - eoversig0 * kT - doversig0 * (kT**2)
                            sig_ratio = doversig0 * (ky0**2) + eoversig0 * ky0 + foversig0
                            field_spectrum_out[j, i - 1] = Ys[i - 1] * (sig_ratio**c_1) * Fky * (gp[j, i - 1] / gamma0) ** (2 * expsub)
                    else:  # SAT2 for electron scale
                        gammaeff = gamma_mix1[j] * (gp[j, i - 1] / gamma0) ** expsub
                        if ky0 > kyetg:
                            gammaeff = gammaeff * np.sqrt(ky0 / kyetg)
                        field_spectrum_out[j, i - 1] = scal * measure * cnorm * ((gammaeff / (kx_width * ky0)) / (1.0 + ay * kx**2)) ** 2
                        if units_in != "GYRO":
                            field_spectrum_out[j, i - 1] = sat_geo_factor * field_spectrum_out[j, i - 1]
                # add these outputs
                gammaeff_out[j, i - 1] = gammaeff
                sig_ratio_out[j, i - 1] = sig_ratio
            kx_width_out[j] = kx_width
            sat_geo_factor_out[j] = sat_geo_factor

    # SAT3 QLA part
    QLA_P = 0.0
    QLA_E = 0.0
    if sat_rule_in == 3:
        QLA_P = np.zeros(nmodes)
        QLA_E = np.zeros(nmodes)
        for k in range(1, nmodes + 1):
            # factor of 2 included for real symmetry
            QLA_P[k - 1] = 2 * mode_transition_function(xs[k - 1], 1.1, 0.6, x_ITG, x_TEM)
            QLA_E[k - 1] = 2 * mode_transition_function(xs[k - 1], 0.75, 0.6, x_ITG, x_TEM)
        QLA_O = 2 * 0.8
    else:
        QLA_P = 1.0
        QLA_E = 1.0
        QLA_O = 1.0

    phinorm = field_spectrum_out
    # so the normal behavior doesn't change,
    if return_phi_params:
        out = dict(
            phinorm=phinorm,
            kx_width=kx_width_out,  # [nky] kx_model (kx rms width)
            gammaeff=gammaeff_out,  # [nky, nmodes] effective growthrate
            kx0_e=kx0_e,  # [nky] spectral shift in kx
            ax=ax,  # SAT1 (cx), SAT2 (alpha_x)
            ay=ay,  # SAT1 (cy)
            exp_ax=exp_ax,  # SAT2 (sigma_x)
            sat_geo_factor=sat_geo_factor_out,  # SAT2=G(theta)**2; SAT1=sat_geo0_out
        )
        if sat_rule_in == 2:
            # add G(theta) params,
            out.update(dict(d1=d1, d2=d2, kycut=kycut, b3=b3))
        if sat_rule_in == 3:
            out.update(dict(sig_ratio=sig_ratio_out, k0=k0))
    else:
        out = phinorm, QLA_P, QLA_E, QLA_O  # SAT123 intensity and QLA params
    return out


def flux_integrals(
    NM,
    NS,
    NF,
    i,
    ky,
    dky0,
    dky1,
    particle,
    energy,
    toroidal_stress,
    parallel_stress,
    exchange,
    particle_flux_out,
    energy_flux_out,
    stress_tor_out,
    stress_par_out,
    exchange_out,
    q_low_out,
    taus_1=1.0,
    mass_2=1.0,
):
    '''
    Compute the flux integrals
    '''
    for nm in range(NM):
        for ns in range(NS):
            for j in range(NF):
                particle_flux_out[nm][ns][j] += dky0 * (0 if i == 0 else particle[i - 1][nm][ns][j]) + dky1 * particle[i][nm][ns][j]
                energy_flux_out[nm][ns][j] += dky0 * (0 if i == 0 else energy[i - 1][nm][ns][j]) + dky1 * energy[i][nm][ns][j]
                stress_tor_out[nm][ns][j] += (
                    dky0 * (0 if i == 0 else toroidal_stress[i - 1][nm][ns][j]) + dky1 * toroidal_stress[i][nm][ns][j]
                )
                stress_par_out[nm][ns][j] += (
                    dky0 * (0 if i == 0 else parallel_stress[i - 1][nm][ns][j]) + dky1 * parallel_stress[i][nm][ns][j]
                )
                exchange_out[nm][ns][j] += dky0 * (0 if i == 0 else exchange[i - 1][nm][ns][j]) + dky1 * exchange[i][nm][ns][j]
            if ky * taus_1 * mass_2 <= 1:
                q_low_out[nm][ns] = energy_flux_out[nm][ns][0] + energy_flux_out[nm][ns][1]
    return particle_flux_out, energy_flux_out, stress_tor_out, stress_par_out, exchange_out, q_low_out


def sum_ky_spectrum(
    sat_rule_in,
    ky_spect,
    gp,
    ave_p0,
    R_unit,
    kx0_e,
    potential,
    particle_QL,
    energy_QL,
    toroidal_stress_QL,
    parallel_stress_QL,
    exchange_QL,
    etg_fact=1.25,
    c0=32.48,
    c1=0.534,
    exp1=1.547,
    cx_cy=0.56,
    alpha_x=1.15,
    **kw,
):
    '''
    Perform the sum over ky spectrum
    The inputs to this function should be already weighted by the intensity function

    nk --> number of elements in ky spectrum
    nm --> number of modes
    ns --> number of species
    nf --> number of fields (1: electrostatic, 2: electromagnetic parallel, 3:electromagnetic perpendicular)

    :param sat_rule_in:

    :param ky_spect: k_y spectrum [nk]

    :param gp: growth rates [nk, nm]

    :param ave_p0: scalar average pressure

    :param R_unit: scalar normalized major radius

    :param kx0_e: spectral shift of the radial wavenumber due to VEXB_SHEAR [nk]

    :param potential: input potential fluctuation spectrum  [nk, nm]

    :param particle_QL: input particle fluctuation spectrum [nk, nm, ns, nf]

    :param energy_QL: input energy fluctuation spectrum [nk, nm, ns, nf]

    :param toroidal_stress_QL: input toroidal_stress fluctuation spectrum [nk, nm, ns, nf]

    :param parallel_stress_QL: input parallel_stress fluctuation spectrum [nk, nm, ns, nf]

    :param exchange_QL: input exchange fluctuation spectrum [nk, nm, ns, nf]

    :param etg_fact: scalar TGLF SAT0 calibration coefficient [1.25]

    :param c0: scalar TGLF SAT0 calibration coefficient [32.48]

    :param c1: scalar TGLF SAT0 calibration coefficient [0.534]

    :param exp1: scalar TGLF SAT0 calibration coefficient [1.547]

    :param cx_cy: scalar TGLF SAT0 calibration coefficient [0.56] (from TGLF 2008 POP Eq.13)

    :param alpha_x: scalar TGLF SAT0 calibration coefficient [1.15] (from TGLF 2008 POP Eq.13)

    :param \**kw: any additional argument should follow the naming convention of the TGLF_inputs

    :return: dictionary with summations over ky spectrum:
            * particle_flux_integral: [nm, ns, nf]
            * energy_flux_integral: [nm, ns, nf]
            * toroidal_stresses_integral: [nm, ns, nf]
            * parallel_stresses_integral: [nm, ns, nf]
            * exchange_flux_integral: [nm, ns, nf]
    '''
    phi_bar_sum_out = 0
    NM = len(energy_QL[0, :, 0, 0])  # get the number of modes
    NS = len(energy_QL[0, 0, :, 0])  # get the number of species
    NF = len(energy_QL[0, 0, 0, :])  # get the number of fields
    particle_flux_out = np.zeros((NM, NS, NF))
    energy_flux_out = np.zeros((NM, NS, NF))
    stress_tor_out = np.zeros((NM, NS, NF))
    stress_par_out = np.zeros((NM, NS, NF))
    exchange_out = np.zeros((NM, NS, NF))
    q_low_out = np.zeros((NM, NS))

    QLA_P = 1
    QLA_E = 1
    QLA_O = 1
    QL_data = np.stack([particle_QL, energy_QL, toroidal_stress_QL, parallel_stress_QL, exchange_QL], axis=4)

    # Multiply QL weights with desired intensity
    if sat_rule_in in [0.0, 0, 'SAT0']:
        intensity_factor = (
            intensity_sat0(ky_spect, gp, ave_p0, R_unit, kx0_e, etg_fact, c0, c1, exp1, cx_cy, alpha_x)
            * potential
            / intensity_sat0(ky_spect, gp, ave_p0, R_unit, kx0_e, 1.25, 32.48, 0.534, 1.547, 0.56, 1.15)
        )
    elif sat_rule_in in [1.0, 1, 'SAT1', 2.0, 2, 'SAT2', 3.0, 3, 'SAT3']:
        intensity_factor, QLA_P, QLA_E, QLA_O = intensity_sat(sat_rule_in, ky_spect, gp, kx0_e, NM, QL_data, **kw)
    elif sat_rule_in in [-1.0, -1, 'DESAT']:
        intensity_factor = (
            intensity_desat(ky_spect, kw['P_PRIME_LOC'], kw['Q_LOC'], kw['TAUS_2'])
            * potential
            / intensity_sat0(ky_spect, gp, ave_p0, R_unit, kx0_e, 1.25, 32.48, 0.534, 1.547, 0.56, 1.15)
        )
    else:
        raise ValueError("sat_rule_in must be [0.0, 0, 'SAT0'] or [1.0, 1, 'SAT1']")

    shapes = [item.shape for item in [particle_QL, energy_QL, toroidal_stress_QL, parallel_stress_QL, exchange_QL] if item is not None][0]

    particle = np.zeros(shapes)
    energy = np.zeros(shapes)
    toroidal_stress = np.zeros(shapes)
    parallel_stress = np.zeros(shapes)
    exchange = np.zeros(shapes)

    for i in range(NS):  # iterate over the species
        for j in range(NF):  # iterate over the fields
            if particle_QL is not None:
                particle[:, :, i, j] = particle_QL[:, :, i, j] * intensity_factor * QLA_P
            if energy_QL is not None:
                energy[:, :, i, j] = energy_QL[:, :, i, j] * intensity_factor * QLA_E
            if toroidal_stress_QL is not None:
                toroidal_stress[:, :, i, j] = toroidal_stress_QL[:, :, i, j] * intensity_factor * QLA_O
            if parallel_stress_QL is not None:
                parallel_stress[:, :, i, j] = parallel_stress_QL[:, :, i, j] * intensity_factor * QLA_O
            if exchange_QL is not None:
                exchange[:, :, i, j] = exchange_QL[:, :, i, j] * intensity_factor * QLA_O

    dky0 = 0
    ky0 = 0
    for i in range(len(ky_spect)):
        ky = ky_spect[i]
        ky1 = ky
        if i == 0:
            dky1 = ky1
        else:
            dky = np.log(ky1 / ky0) / (ky1 - ky0)
            dky1 = ky1 * (1.0 - ky0 * dky)
            dky0 = ky0 * (ky1 * dky - 1.0)

        particle_flux_out, energy_flux_out, stress_tor_out, stress_par_out, exchange_out, q_low_out = flux_integrals(
            NM,
            NS,
            NF,
            i,
            ky,
            dky0,
            dky1,
            particle,
            energy,
            toroidal_stress,
            parallel_stress,
            exchange,
            particle_flux_out,
            energy_flux_out,
            stress_tor_out,
            stress_par_out,
            exchange_out,
            q_low_out,
        )
        ky0 = ky1
        results = {
            "particle_flux_integral": particle_flux_out,
            "energy_flux_integral": energy_flux_out,
            "toroidal_stresses_integral": stress_tor_out,
            "parallel_stresses_integral": stress_par_out,
            "exchange_flux_integral": exchange_out,
        }
    return results


import xarray as xr

specnames = ["e", "D", "C"]


def calculate_QLGYRO_flux(runid, cgyro_freq, cgyro_ql, tglf_growth, sat_rule_in, alpha_zf_in, in0):
    # Calculate the QL flux from SAT1 using the quasilinear weights
    # derived from CGYRO. This will give the same heat flux as TGLF if the QL
    # weights are the same, and will show the effect of the differences in QL weights

    # gammas = asarray(tglf['eigenvalue_spectrum']['gamma']).T
    # CGYRO inputs
    cgyro_kys = cgyro_freq.sel(dim="ky").data
    growth = cgyro_freq.sel(dim="gamma").data
    # tglf_growth_rates = tglf_growth.sel(dim="gamma").data
    cgyro_gammas = np.array([growth, growth * 0]).T
    # tglf_gammas = np.array([tglf_growth_rates,tglf_growth_rates*0])

    # data = OMFIT["scratch"]["linear_tglf_cgyro"]
    # index = np.where(OMFIT["scratch"][dbname]["id"] == runid)[0][0]
    # print("index", index)
    # print("fn", OMFIT["scratch"][dbname]["fn"][index], "sat:", sat_rule_in, "azf: ", alpha_zf_in)
    # for key, value in data.items():
    #     if isinstance(value, np.ndarray) or isinstance(value, xr.DataArray):
    #         if value.ndim == 1 and not key.endswith("profile"):
    #             in0[key] = value[index]
    in0["ALPHA_E"] = 1
    in0["RLNP_CUTOFF"] = 18
    in0["ALPHA_QUENCH"] = 0
    in0["ALPHA_ZF"] = alpha_zf_in
    in0["TAUS_1"] = 1
    in0["AS_1"] = 1
    in0["SAT_RULE"] = sat_rule_in
    if sat_rule_in == 1:
        in0["UNITS"] = "GYRO"
    else:
        in0["UNITS"] = "CGYRO"
    gammas = np.array([growth, growth * 0])

    rho_ion = 0.0
    charge = 0.0

    if not bool(in0["USE_AVE_ION_GRID"]):
        rho_ion = (in0["MASS_2"] * in0["TAUS_2"]) ** 0.5 / in0["ZS_2"]
    else:
        for is_ in range(1, in0["NS"] + 1):  # Assuming N is the upper limit of the loop
            if is_ > 1 and (in0["ZS_%s" % is_] * in0["AS_%s" % is_]) / abs(in0["AS_1"] * in0["ZS_1"]) > 0.1:
                charge += in0["ZS_%s" % is_] * in0["AS_%s" % is_]
                rho_ion += (
                    in0["ZS_%s" % is_]
                    * in0["AS_%s" % is_]
                    * (in0["MASS_%s" % is_] * in0["TAUS_%s" % is_]) ** 0.5
                    / in0["ZS_%s" % is_]
                )
        rho_ion /= charge

    in0["rho_ion"] = rho_ion
    in0["SAT_geo0_out"] = 1
    kx0_e, satgeo1, satgeo2, R_unit, bt0, bgeo0, gradr0, _, _, _, _ = get_sat_params(
        sat_rule_in, cgyro_kys, gammas, **in0
    )

    in0["SAT_geo1_out"] = satgeo1
    in0["SAT_geo2_out"] = satgeo2
    in0["B_geo0_out"] = bgeo0
    in0["Bt0_out"] = bt0
    in0["grad_r0_out"] = gradr0

    # Add empty mode dimension to bring QL data into TGLF format:
    QL_data = cgyro_ql.expand_dims({"mode": 2}, axis=[1]).copy()
    # Set subdominant mode to zero
    QL_data.loc[dict(mode=1)].data.fill(0)
    # Need to reorder ions from D, C, e- to e-, D, C:
    QL_data = QL_data.roll(species=1, roll_coords=True)

    cgyro_particle_QL = QL_data.sel(moment="particle")
    cgyro_energy_QL = QL_data.sel(moment="energy")
    cgyro_stresstor_QL = QL_data.sel(moment="momentum")
    # sign of BT_EXP affects sign of gB normalization in CGYRO for QL weights
    sign_Bt = in0["BT_EXP"] / abs(in0["BT_EXP"])
    cgyro_particle_QL *= sign_Bt
    cgyro_energy_QL *= sign_Bt
    cgyro_stresstor_QL *= sign_Bt

    QL_data = np.stack(
        [
            cgyro_particle_QL,
            cgyro_energy_QL,
            cgyro_stresstor_QL,
            np.zeros(cgyro_stresstor_QL.shape),
            np.zeros(cgyro_stresstor_QL.shape),
        ],
        axis=4,
    )

    intensity_cgyro, QLA_P, QLA_E, QLA_O = intensity_sat(sat_rule_in, cgyro_kys, cgyro_gammas, kx0_e, 1, QL_data, **in0)
    heat_flux_spectrum = np.sum(np.sum(cgyro_energy_QL, axis=1), axis=1) * intensity_cgyro * QLA_E
    heat_flux_spectrum = np.sum(heat_flux_spectrum, axis=1)  # sum over fields


    cgyro_sat = sum_ky_spectrum(
        sat_rule_in,  #
        cgyro_kys,  #
        cgyro_gammas,  #
        np.zeros((cgyro_kys).shape),  # avep0 - only needed for SAT0
        R_unit,  #
        kx0_e,  # spectral shift
        np.zeros((cgyro_gammas).shape),  # potential  = intensity
        cgyro_particle_QL,
        cgyro_energy_QL,
        cgyro_stresstor_QL,
        np.zeros((cgyro_stresstor_QL).shape),
        np.zeros((cgyro_stresstor_QL).shape),
        **in0,
    )

    
    print(cgyro_sat.keys())
    # cgyro_sat['energy_flux_integral'] dimensions are: nmodes, ns, nfield
    cgyro_satG = np.sum(np.sum(cgyro_sat["particle_flux_integral"], axis=2), axis=0)
    cgyro_satQ = np.sum(np.sum(cgyro_sat["energy_flux_integral"], axis=2), axis=0)
    cgyro_satP = np.sum(np.sum(cgyro_sat["toroidal_stresses_integral"], axis=2), axis=0)
    flux_matrix = np.stack([cgyro_satG, cgyro_satQ, cgyro_satP], axis=1)  # shape: (3, 3)
    # now start converting to the interested fluxes: Ge,Qe,Qi,Pi, per wavenumber
    G_elec_per_ky = flux_matrix[0, 0]  # (size,nky,)
    Q_elec_per_ky = flux_matrix[0, 1]  # (size,nky,)
    Q_ions_per_ky = np.sum(flux_matrix[1:, 1], axis=0)
    P_ions_per_ky = np.sum(flux_matrix[1:, 2], axis=0)
    # cat them tobe (size, nky, 4)
    return (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky)


    


import os
import numpy as np
from omfit_classes.omfit_gacode import OMFITcgyro
from xarray import DataArray

def load_cgyro_scan(root_dir):
    # Step 1: Collect subdirectories representing different ky points
    def extract_ky_index(name):
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return float("inf")  # Push malformed names to the end

    ky_dirs = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("ky_")
    ], key=lambda d: extract_ky_index(os.path.basename(d)))
    ql_flux_list = []
    ky_vals = []
    gamma_vals = []
    omega_vals = []
    species_names = None
    bt_sign = None

    # Step 2: Load each CGYRO run
    for d in ky_dirs:
        gyro = OMFITcgyro(d)
        gyro.load()

        if 'freq' not in gyro:
            print(f"⚠️ Skipping {d}: missing 'freq'")
            continue

        gamma_arr = gyro['freq']['gamma']
        omega_arr = gyro['freq']['omega']
        
        # Take the last time value at ky=0
        gamma = gamma_arr.isel(ky=0, t=-1).item()
        omega = omega_arr.isel(ky=0, t=-1).item()

        if gamma <= 0:
            print("parsed due to neg gamma")
            continue  # Skip stable or decaying modes


        gamma_vals.append(gamma)
        omega_vals.append(omega)

        # Get ky value as scalar
        ky = gamma_arr.coords["ky"].values[0]
        # Extract the last time slice for each moment and stack them
        qlflux_moments = []
        for moment in ['particle', 'energy', 'momentum']:
            qlflux_timed = gyro['qlflux_ky'][moment]  # shape: (species, field, t)
            qlflux_last = qlflux_timed.isel(t=-1).data  # shape: (species, field)
            qlflux_moments.append(qlflux_last)

        # Stack into shape (species, field, moment)
        qlflux = np.stack(qlflux_moments, axis=-1)

        # Scale by ky and append
        ql_flux_list.append(qlflux * ky)
        ky_vals.append(ky)

        # Save metadata once
        if species_names is None:
            species_names = list(gyro['species_tags'])

        if bt_sign is None and 'units' in gyro:
            bt = gyro['units'].get('bt0', 1.0)
            bt_sign = bt / abs(bt)

    return {
        "ql_flux": np.array(ql_flux_list),
        "ky": np.array(ky_vals) * -1,
        "gamma": np.array(gamma_vals),
        "omega": np.array(omega_vals),
        "species_names": species_names,
        "bt_sign": bt_sign
    }

def package_as_dataarray(ql_flux, ky_vals, species_names, bt_sign):
    nky = len(ky_vals)
    nsp, nfield, nmoment = ql_flux.shape[1:]

    ql_flux *= bt_sign  # apply Bt sign correction

    da = DataArray(
        data=ql_flux,
        dims=["ky", "species", "field", "moment"],
        coords={
            "ky": ky_vals,
            "species": species_names,
            "field": ["Phi", "Apar", "Bpar"],
            "moment": ["particle", "energy", "momentum"],
        },
        attrs={"note": "Quasilinear flux scaled by ky and Bt sign"}
    )
    return da


import os
import numpy as np
import xarray as xr

def parse_tglf_input_file(filepath):
    in0 = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines and comments
            if '=' not in line:
                continue

            try:
                key, val = map(str.strip, line.split("="))
                if val.lower() in ("true", ".true."):
                    in0[key] = True
                elif val.lower() in ("false", ".false."):
                    in0[key] = False
                else:
                    in0[key] = float(val)
            except Exception as e:
                print(f"⚠️ Failed to parse line: {line} -- {e}")
                continue

    # Convert NS to int explicitly
    if "NS" in in0:
        in0["NS"] = int(in0["NS"])
    return in0

import os
import numpy as np
import xarray as xr
from collections import defaultdict

def run_qlgyro_flux_from_dir(
    root_dir,
    tglf_input_path,
    sat_rules=(2, 3),
    alpha_zf_in=1.0
):
    # Step 1: Load CGYRO outputs
    cgyro_data = load_cgyro_scan(root_dir)
    print(cgyro_data["ky"])
    if len(cgyro_data["ky"]) == 0:
        print("❌ No valid CGYRO runs found.")
        return {}

    # Step 2: Package into xarray
    ql_data = package_as_dataarray(
        ql_flux=cgyro_data["ql_flux"],
        ky_vals=cgyro_data["ky"],
        species_names=cgyro_data["species_names"],
        bt_sign=cgyro_data["bt_sign"]
    )

    # Step 3: Prepare input dictionary from TGLF input
    in0 = parse_tglf_input_file(tglf_input_path)
    bt_exp = in0.get("BT_EXP", 1.0)

    # Patch BT_EXP into freq container
    from xarray import DataArray
    cgyro_freq = DataArray(
        data=np.stack([cgyro_data["ky"], cgyro_data["gamma"], cgyro_data["omega"]], axis=1),
        dims=["ky", "dim"],
        coords={"ky": cgyro_data["ky"], "dim": ["ky", "gamma", "omega"]},
        attrs={"BT_EXP": bt_exp}
    )

    # Use CGYRO gammas for now as a placeholder for tglf_growth
    tglf_growth = cgyro_freq.sel(dim="gamma")

    results = defaultdict(list)

    for sat_rule in sat_rules:
        print(f"\n▶ Running QLGYRO with SAT{sat_rule}")
        (
            f1, f2, f3, f4
        ) = calculate_QLGYRO_flux(
            runid="cgyro_dir",
            cgyro_freq=cgyro_freq,
            cgyro_ql=ql_data,
            tglf_growth=tglf_growth,
            sat_rule_in=sat_rule,
            alpha_zf_in=alpha_zf_in,
            in0 = in0
        )
    print(f1, f2, f3, f4)
    return results

def load_cgyro_scan_with_ky_flux(root_dir):
    # This is only for non-linear runs.
    def extract_ky_index(name):
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return float("inf")

    ky_dirs = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("ky_")
    ], key=lambda d: extract_ky_index(os.path.basename(d)))

    flux_per_ky = []
    ky_vals = []
    gamma_vals = []
    omega_vals = []
    species_names = None
    bt_sign = None

    for d in ky_dirs:
        gyro = OMFITcgyro(d)
        gyro.load()
        with open(gyro['out.cgyro.info'].filename) as runtext:
            lines = runtext.readlines()
        status = ''
        for line in lines:
            print(line)
            if 'EXIT' in line:
                status = line.split(':')[1].strip()
        print(status)
        if 'converged' in status:
            converged = True
        else:
            converged = False
        print(converged, "HAIYAAAAA")
        if "freq" not in gyro or "flux_ky" not in gyro:
            continue

        gamma_arr = gyro["freq"]["gamma"]
        omega_arr = gyro["freq"]["omega"]
        gamma = gamma_arr.isel(ky=0, t=-1).item()
        omega = omega_arr.isel(ky=0, t=-1).item()
        # if gamma <= 0:
        #     print(f"⏭️ Skipped {d} due to non-positive gamma: {gamma}")
        #     continue

        gamma_vals.append(gamma)
        omega_vals.append(omega)

        ky = gamma_arr.coords["ky"].values[0]
        ky_vals.append(ky)

        qlflux_moments = []
        for moment in ["particle", "energy", "momentum"]:
            qlflux_timed = gyro["flux_ky"][moment]  # (species, field, t)
            qlflux_last = qlflux_timed.isel(t=-1).data  # (species, field)
            qlflux_moments.append(qlflux_last)  # append (species, field)

        # Stack into (species, field, moment)
        flux_tensor = np.stack(qlflux_moments, axis=-1)  # (species, field, moment)
        flux_per_ky.append(flux_tensor)

        if species_names is None:
            species_names = list(gyro["species_tags"])
        if bt_sign is None and "units" in gyro:
            bt = gyro["units"].get("bt0", 1.0)
            bt_sign = bt / abs(bt)

    if not flux_per_ky:
        print("❌ No valid CGYRO runs found.")
        return {}

    flux_per_ky = np.stack(flux_per_ky, axis=-1)  # (species, field, moment, nky)
    flux_matrix = flux_per_ky.squeeze()  # alias
    # === Compute Flux Spectra ===
    G_elec_per_ky = np.sum(flux_matrix[0, 0, 0, :], axis=0)         # species 0, field 0, particle
    Q_elec_per_ky = np.sum(flux_matrix[0, :, 1, :], axis=0)         # species 0, all fields, energy
    Q_ions_per_ky = np.sum(np.sum(flux_matrix[1:, :, 1, :], axis=1), axis=0)  # ions, all fields, energy
    P_ions_per_ky = np.sum(np.sum(flux_matrix[1:, :, 2, :], axis=1), axis=0)  # ions, all fields, momentum

    # === Total Fluxes ===
    total_fluxes = np.array([
        np.sum(G_elec_per_ky),
        np.sum(Q_elec_per_ky),
        np.sum(Q_ions_per_ky),
        np.sum(P_ions_per_ky),
    ])

    return {
        "spectra": {
            "G_elec_per_ky": G_elec_per_ky,
            "Q_elec_per_ky": Q_elec_per_ky,
            "Q_ions_per_ky": Q_ions_per_ky,
            "P_ions_per_ky": P_ions_per_ky,
        },
        "total_fluxes": total_fluxes,  # shape (4,)
        "ky": np.array(ky_vals) * -1,
        "gamma": np.array(gamma_vals),
        "omega": np.array(omega_vals),
        "species_names": species_names,
        "bt_sign": bt_sign,
    }



if __name__ == "__main__":
    cgyro_scan = "/global/homes/w/wyl002/Github/ai-fusion-gknn-wesley/tglf_outputs/tglf_input_20250720_145102/sample_4"
    input_tglf_file = "/global/homes/w/wyl002/Github/ai-fusion-gknn-wesley/tglf_outputs/tglf_input_20250720_145102/sample_4/ky_0/input.tglf"
    # data = load_cgyro_scan_with_ky_flux(cgyro_scan)
    results = run_qlgyro_flux_from_dir(
    root_dir=cgyro_scan,
    tglf_input_path=input_tglf_file,
    sat_rules=(2, 3),
    alpha_zf_in=1.0
    )
