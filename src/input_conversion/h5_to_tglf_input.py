import h5py
import numpy as np
import os
from datetime import datetime
from pyrokinetics import Pyro
# Keys to extract from HDF5


{'NN_MAX_ERROR': -1.0, 
 'MASS_2': 1.0, 
 'THETA0_SA': 0.0, 
 'NXGRID': 16, 
 'XWELL_SA': 0.0, 
 'VPAR_SHEAR_MODEL': 1, 
 'VNS_SHEAR_1': 0.0, 
 'VNS_SHEAR_2': 0.0, 
 'SHAT_SA': 1.0, 
 'RLNP_CUTOFF': 18.0, 
 'WIDTH_MIN': 0.3, 
 'NBASIS_MIN': 2, 
 'USE_AVE_ION_GRID': True, 
 'FT_MODEL_SA': 1, 
 'MASS_1': 0.0002723125672605524, 
 'GEOMETRY_FLAG': 1, 
 'NMODES': 5,
  'WIDTH': 1.65, 
  'VEXB': 0.0, 
  'VPAR_MODEL': 0, 
  'VTS_SHEAR_3': 0.0, 
  'FIND_WIDTH': True, 
  'FILTER': 2.0, 
  'DAMP_SIG': 0.0, 
  'SIGN_BT': -1, 
  'LINSKER_FACTOR': 0.0, 
  'ZS_3': 6.0, 
  'ALPHA_QUENCH': 0, 
  'MASS_3': 6.0, 
  'TAUS_1': 1.0, 
  'USE_BISECTION': True, 'DAMP_PSI': 0.0, 'ZS_2': 1.0, 
  'WRITE_WAVEFUNCTION_FLAG': 0, 'THETA_TRAPPED': 0.7, 
  'GHAT': 1.0, 'VTS_SHEAR_2': 0.0, 'WD_ZERO': 0.1, 
  'AS_1': 1.0, 'rho_e': 0.01650189586867377, 
  'SAT_RULE': 2, 'ALPHA_ZF': 1.0, 'NEW_EIKONAL': True, 
  'ADIABATIC_ELEC': False, 'DEBYE_FACTOR': 1.0, 'VTS_SHEAR_1': 0.0,
   'B_unit': 1.0, 'ALPHA_SA': 0.0, 'DRMINDX_LOC': 1.0, 'KY': 0.3, 
   'GRADB_FACTOR': 0.0, 'RMIN_SA': 0.5, 'ALPHA_MACH': 0.0, 'NWIDTH': 21, 
   'SAT_geo0_out': 1.0, 'GCHAT': 1.0, 'USE_MHD_RULE': False, 'RMAJ_SA': 3.0, 
   'IBRANCH': -1, 'USE_TRANSPORT_MODEL': True, 'VNS_SHEAR_3': 0.0, 'SIGN_IT': 1, 
   'NBASIS_MAX': 6, 'USE_INBOARD_DETRAPPED': False, 'ZS_1': -1.0, 'USE_BPER': True, 
   'WDIA_TRAPPED': 0.0, 'NKY': 17, 'KYGRID_MODEL': 4, 'UNITS': 'CGYRO', 'USE_BPAR': False,
     'KX0_LOC': 0.0, 'Q_SA': 2.0, 'B_MODEL_SA': 1, 'ETG_FACTOR': 1.25, 'NS': 3, 'IFLUX': True,
       'PARK': 1.0, 'ALPHA_E': 1.0, 'ALPHA_P': 1.0, 'XNU_FACTOR': 1.0}
input_keys = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1",
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1",
    "RLTS_2", "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3",
    "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR",
    "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10", "DEBYE_log10"
]

log_ops_keys = {"BETAE_log10", "XNUE_log10", "DEBYE_log10"}

avg_ky_locs = [0.06010753, 0.12021505, 0.18032258, 0.2404301, 0.30053763, 0.54096774,
  0.66118279, 0.78139784, 0.90161289, 1.02182795, 1.142043, 1.26225805,
  1.20215052, 1.5988144, 2.12677655, 2.82962789, 3.76547314, 5.01177679,
  6.67183152, 8.88339377, 11.8302146, 15.75743919, 20.99217655, 27.97098031]

import os
import h5py
from datetime import datetime
from pyrokinetics import Pyro
import numpy as np
import textwrap

# Fixed trailer with KY dynamically inserted
FIXED_TRAILER_TEMPLATE = textwrap.dedent("""\
    GEOMETRY_FLAG = 1
    SIGN_BT=-1.00000E+00
    SIGN_IT=+1.00000E+00

    #----------Additional Parameters----------
    # Species
    NS=3
    N_MODES=5
    # Questionable forced defaults:
    DRMINDX_LOC=1.0
    NKY=1
    USE_BPER=True
    USE_BPAR=True
    USE_AVE_ION_GRID=True
    USE_MHD_RULE=False
    ALPHA_ZF=-1
    KYGRID_MODEL=0
    KY={ky_val}
    SAT_RULE=3
    NBASIS_MAX=6
    UNITS=CGYRO
    VPAR_2 = 0.0
    VPAR_3 = 0.0

    VPAR_SHEAR_2 = 0.0
    VPAR_SHEAR_3 = 0.0

    #Confirmed with Tom 7/19
    AS_1=+1.0
    TAUS_1=+1.0
    MASS_1=0.0002723125672605524
    ZS_1=-1
    MASS_2=+1.0
    ZS_2=1
    MASS_3=+6.0
    ZS_3=6.0
""")

CGYRO_OVERRIDE_VALUES = {
    'N_ENERGY':8,
    'N_XI':24,
    'N_THETA':24,
    'N_RADIAL':16,
    'N_TOROIDAL':1,
    'NONLINEAR_FLAG':0,
    'BOX_SIZE':1,
    'DELTA_T':0.005,
    'MAX_TIME':100000.0,
    'PRINT_STEP':100,
    'THETA_PLOT': 1
}

def write_input_tglf(f, sample_idx, ky_idx, out_dir):
    with open(os.path.join(out_dir, "input.tglf"), "w") as f_out:
        f_out.write("# Geometry (Miller) and Parameters\n")
        for key in input_keys:
            val = f[key][sample_idx]
            if key in log_ops_keys:
                val = 10 ** val
                key_out = key.replace("_log10", "")
            else:
                key_out = key

            f_out.write(f"{key_out}={val:+.5E}\n")

        ky_val = f["ky"][sample_idx, ky_idx]
        trailer = FIXED_TRAILER_TEMPLATE.format(ky_val=f"{ky_val:+.5E}")
        f_out.write("\n" + trailer + "\n")

def generate_tglf_and_cgyro(f, sample_idx, ky_idx, out_dir):
    write_input_tglf(f, sample_idx, ky_idx, out_dir)
    pyro = Pyro(gk_file=os.path.join(out_dir, "input.tglf"))
    for species in pyro.local_species['names']:
        pyro.local_species.enforce_quasineutrality(species)
        break
    pyro.write_gk_file(os.path.join(out_dir, "input.cgyro"), gk_code="CGYRO", enforce_quasineutrality=True)

def convert_h5_to_tglf_dirs(h5_path, out_root="tglf_outputs"):
    os.makedirs(out_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = os.path.join(out_root, f"tglf_input_{timestamp}")
    os.makedirs(main_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n_samples, n_ky = f["ky"].shape
        for sample_idx in range(n_samples):
            sample_dir = os.path.join(main_dir, f"sample_{sample_idx}")
            os.makedirs(sample_dir, exist_ok=True)

            for ky_idx in range(n_ky):
                ky_dir = os.path.join(sample_dir, f"ky_{ky_idx}")
                os.makedirs(ky_dir, exist_ok=True)
                generate_tglf_and_cgyro(f, sample_idx, ky_idx, ky_dir)

    print(f"✅ Done. TGLF input files written to: {main_dir}")

def override_cgyro_numerics(f_path, ky_val):
    overrides = CGYRO_OVERRIDE_VALUES
    if ky_val > 1 and ky_val <= 10:
        # Non ion-scale, finer timesteps and less frequent printing
        overrides['DELTA_T'] = 0.001
        overrides['PRINT_STEP'] = 250
    elif ky_val > 10:
        # Much finer timesteps, very infrequent printing
        overrides['DELTA_T'] = 0.0005
        overrides['PRINT_STEP'] = 500

    with open(f_path, 'r') as f:
        lines = f.readlines()
        f.close()
    newlines = []
    for line in lines:
        foundKey = False
        for key in overrides:
            if key in line:
                newline = f'{key} = {overrides[key]}\n'
                foundKey = True
                break
        if not foundKey:
            newline = line
        newlines.append(newline)

    with open(f_path, 'w') as f:
        for line in newlines:
            f.write(line)
        f.close()
    return

def write_input_tglf_from_arr(input_arr, ky_val, out_dir):
    with open(os.path.join(out_dir, "input.tglf"), "w") as f_out:
        f_out.write("# Geometry (Miller) and Parameters\n")
        for i in range(len(input_keys)):
            key = input_keys[i]
            val = input_arr[i]
            if key in log_ops_keys:
                val = 10 ** val
                key_out = key.replace("_log10", "")
            else:
                key_out = key

            f_out.write(f"{key_out}={val:+.5E}\n")

        trailer = FIXED_TRAILER_TEMPLATE.format(ky_val=f"{ky_val:+.5E}")
        f_out.write("\n" + trailer + "\n")

def generate_tglf_and_cgyro_from_arr(input_arr, ky_val, out_dir, species_to_enforce_qn='ion2'):
    write_input_tglf_from_arr(input_arr, ky_val, out_dir)
    pyro = Pyro(gk_file=os.path.join(out_dir, "input.tglf"))
    # ion2 corresponds to the subdominant ion (Carbon)
    pyro.local_species.enforce_quasineutrality(species_to_enforce_qn)
    f_path = os.path.join(out_dir, "input.cgyro")
    pyro.write_gk_file(f_path, gk_code="CGYRO", enforce_quasineutrality=True)
    override_cgyro_numerics(f_path, ky_val)

def convert_numpy_to_tglf_dirs(samples, n_ky, out_root="tglf_outputs"):
    os.makedirs(out_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = os.path.join(out_root, f"tglf_input_{timestamp}")
    os.makedirs(main_dir, exist_ok=True)
    
    n_samples = samples.shape[0]
    
    for sample_idx in range(n_samples):
        sample_dir = os.path.join(main_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        sample = samples[sample_idx]
        for ky_idx in range(n_ky):
            ky_val = avg_ky_locs[ky_idx]
            ky_dir = os.path.join(sample_dir, f"ky_{ky_idx}")
            os.makedirs(ky_dir, exist_ok=True)
            generate_tglf_and_cgyro_from_arr(sample, ky_val, ky_dir)

    print(f"✅ Done. TGLF input files written to: {main_dir}")