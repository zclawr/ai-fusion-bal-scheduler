import os
import numpy as np
from omfit_classes.omfit_gacode import OMFITcgyro, OMFITgyro, OMFITinputgacode

def load_cgyro_simulations(root_dir):
    """
    Load each CGYRO simulation from subdirectories under root_dir.
    Returns a list of tuples: (OMFITcgyro object, directory name, ky, gamma)
    """
    sims = []
    for sub in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, sub)
        if not os.path.isdir(d):
            continue
        try:
            gyro = OMFITcgyro(d)
            gyro.load()
            ky = gyro['freq']['gamma'].coords['ky'].values.item()
            gamma = gyro['freq']['gamma'].isel(t=-1).values.item()
            sims.append((gyro, sub, ky, gamma))
        except Exception as e:
            print(f"Skipping {d} due to error: {e}")
    return sims

def evaluate_fluxes_from_simulations(sims, w=0.2, u='gB', n=None):
    """
    For each loaded CGYRO simulation, compute fluxes using get_fluxes().
    Returns: list of (ky, gamma, flux matrix) where flux matrix = (n_species, 3)
    """
    flux_data = []
    for gyro, name, ky, gamma in sims:
        if gamma <= 0:
            print(f"Skipping ky={ky:.3f} due to gamma <= 0")
            continue
        try:
            outputs = {'cgyro': gyro}
            fluxes, tags, factors = get_fluxes(outputs=outputs, w=w, u=u, n=n)
            flux_data.append((ky, gamma, fluxes))
        except Exception as e:
            print(f"Error processing {name} (ky={ky}): {e}")
    return flux_data, tags

def run_converged(outputs):
    """
    This script determines if a run is converged
    """
    converged = False

    if 'gyro' in outputs and isinstance(outputs['gyro'], OMFITgyro) and 'out.gyro.run' in outputs['gyro']:
        with open(outputs['gyro']['out.gyro.run'].filename) as runtext:
            lines = runtext.readlines()
        status = ''
        for line in lines:
            if 'STATUS' in line:
                status = line.split(':')[1].strip()

        if status == 'converged':
            converged = True
        else:
            converged = False

    if 'cgyro' in outputs and isinstance(outputs['cgyro'], OMFITcgyro) and 'out.cgyro.info' in outputs['cgyro']:
        with open(outputs['cgyro']['out.cgyro.info'].filename) as runtext:
            lines = runtext.readlines()
        status = ''
        for line in lines:
            if 'EXIT' in line:
                status = line.split(':')[1].strip()

        if 'converged' in status:
            converged = True
        else:
            converged = False

    return converged

def get_fluxes(outputs, w=0.2, u='gB', n=None, sumfields=True):
    """
    Get or calculate the quasi-linear or nonlinear transport flux for particles, energy and momentum
    """

    # Determine if this run is converged (general to GYRO and CGYRO)
    converged = run_converged(outputs=outputs)

    # Determine the code
    if 'gyro' in outputs and isinstance(outputs['gyro'], OMFITgyro) and 'nonlinear_run' in outputs['gyro']:
        code = 'gyro'
    if 'cgyro' in outputs and isinstance(outputs['cgyro'], OMFITcgyro) and 'nonlinear_run' in outputs['cgyro']:
        code = 'cgyro'

    # Determine if the run is linear or nonlinear
    if outputs[code]['nonlinear_run']:
        if code == 'gyro':
            flux_key = 'flux_r'
        else:
            flux_key = 'flux_t'
    else:
        flux_key = 'qlflux_ky'

    # Determine the number of species
    if code == 'gyro':
        n_species = outputs['gyro']['profile']['n_kinetic']
    elif code == 'cgyro':
        n_species = outputs['cgyro']['n_species']

    # Set the names of the moments
    moments = ['particle', 'energy', 'momentum']

    # Pre-allocate the fluxes
    if sumfields:
        fluxes = np.zeros((n_species, len(moments)))
    else:
        fluxes = np.zeros((n_species, 3, len(moments)))

    # Set the units and tags
    if u == 'gB':
        unit = np.repeat(1.0, 3)

        if n is None:
            tagmoms = ['\Gamma/\Gamma_{gB}', 'Q/Q_{gB}', '\Pi/\Pi_{gB}']
        elif n == 'Qe':
            tagmoms = ['(\Gamma/\Gamma_{gB})/(Q_e/Q_{gB})', '(Q/Q_{gB})/(Q_e/Q_{gB})', '(\Pi/\Pi_{gB})/(Q_e/Q_{gB})']
        elif n == 'Qi':
            tagmoms = ['(\Gamma/\Gamma_{gB})/(Q_i/Q_{gB})', '(Q/Q_{gB})/(Q_i/Q_{gB})', '(\Pi/\Pi_{gB})/(Q_i/Q_{gB})']
    elif u == 'MKS':
        if code == 'gyro':
            unit = np.array(
                [outputs['gyro']['units']['Gamma_gBD'] * 0.624 * 1e3, outputs['gyro']['units']['Q_gBD'], outputs['gyro']['units']['Pi_gBD']]
            )
        elif code == 'cgyro':
            unit = np.array(
                [
                    outputs['cgyro']['units']['gamma_gb_norm'],
                    outputs['cgyro']['units']['q_gb_norm'],
                    outputs['cgyro']['units']['pi_gb_norm'],
                ]
            )

        if n is None:
            tagmoms = ['\Gamma(e19/m^2/s)', 'Q(MW/m^2)', '\Pi(Nm/m^2)']
        elif n == 'Qe':
            tagmoms = ['\Gamma(e19/s)/Q_e(MW)', 'Q/Q_e', '\Pi(Nm)/Q_e(MW)']
        elif n == 'Qi':
            tagmoms = ['\Gamma(e19/s)/Q_i(MW)', 'Q/Q_i', '\Pi(Nm)/Q_i(MW)']

    # Compute the fluxes in gB units
    if sumfields:
        for imom, mom in enumerate(moments):
            for ispec in range(n_species):
                if outputs[code]['nonlinear_run']:
                    fluxes[ispec, imom] = (
                        outputs[code][flux_key][mom]
                        .sum(dim='field')
                        .isel(species=ispec)
                        .where(outputs[code][flux_key][mom]['t'] >= (1.0 - w) * outputs[code][flux_key][mom]['t'].max(), drop=True)
                        .mean()
                        .values
                    )
                else:
                    if converged:
                        fluxes[ispec, imom] = outputs[code][flux_key][mom].sum(dim='field').isel(species=ispec).values[-1]
                    else:
                        fluxes[ispec, imom] = (
                            outputs[code][flux_key][mom]
                            .sum(dim='field')
                            .isel(species=ispec)
                            .where(outputs[code][flux_key][mom]['t'] >= (1.0 - w) * outputs[code][flux_key][mom]['t'].max(), drop=True)
                            .mean()
                            .values
                        )
    else:
        for f in range(3):
            if f < 2:
                for imom, mom in enumerate(moments):
                    for ispec in range(n_species):
                        if outputs[code]['nonlinear_run']:
                            fluxes[ispec, imom] = (
                                outputs[code][flux_key][mom]
                                .isel(species=ispec)
                                .where(outputs[code][flux_key][mom]['t'] >= (1.0 - w) * outputs[code][flux_key][mom]['t'].max(), drop=True)
                                .mean()
                                .values
                            )
                        else:
                            if converged:
                                fluxes[ispec, f, imom] = outputs[code][flux_key][mom].isel(species=ispec, field=f).values[-1]
                            else:
                                fluxes[ispec, f, imom] = (
                                    outputs[code][flux_key][mom]
                                    .isel(species=ispec, field=f)
                                    .where(
                                        outputs[code][flux_key][mom]['t'] >= (1.0 - w) * outputs[code][flux_key][mom]['t'].max(), drop=True
                                    )
                                    .mean()
                                    .values
                                )

    # Perform the unit and normalization across the moment dimension
    if n == 'Qe':
        # If we normalize by Qe then we compute Qe in desired units
        # Then our factor is the units for all fluxes / Qe in those units
        # This makes Qe = 1 in chosen units
        Qe = fluxes[-1, 1] * unit[1]
        factor = unit / Qe
    elif n == 'Qi':
        Qi = fluxes[0, 1] * unit[1]
        factor = unit / Qi
    else:
        factor = unit

    # Apply along the flux axis for all species
    fluxes *= factor[np.newaxis, :]

    return fluxes, tagmoms, factor

# Update the script to include the final flux matrix computation

def compute_flux_matrix(flux_data):
    """
    Given a list of (ky, gamma, flux) tuples, return:
    - flux_per_ky: np.ndarray of shape (nky, ns, 3)
    - target_flux_per_ky: np.ndarray of shape (nky, 4)
    """
    # Sort by ky
    flux_data_sorted = sorted(flux_data, key=lambda x: x[0])
    ky_sorted = [item[0] for item in flux_data_sorted]
    fluxes_sorted = [item[2] for item in flux_data_sorted]

    # Stack into (nky, ns, 3)
    flux_per_ky = np.stack(fluxes_sorted, axis=0)

    # Compute the derived fluxes per ky
    G_elec_per_ky = flux_per_ky[:, 0, 0]               # electrons, particle
    Q_elec_per_ky = flux_per_ky[:, 0, 1]               # electrons, energy
    Q_ions_per_ky = np.sum(flux_per_ky[:, 1:, 1], axis=-1)  # ions, energy
    P_ions_per_ky = np.sum(flux_per_ky[:, 1:, 2], axis=-1)  # ions, momentum

    # Stack into (nky, 4)
    target_flux_per_ky = np.stack(
        (G_elec_per_ky, Q_elec_per_ky, Q_ions_per_ky, P_ions_per_ky),
        axis=-1
    )

    return np.array(ky_sorted), flux_per_ky, target_flux_per_ky

# Example usage:
# ky_sorted, flux_matrix, target_flux_matrix = compute_flux_matrix(flux_data)



import os
import h5py
import numpy as np

def main():
    root_dir = "/global/homes/w/wyl002/Github/ai-fusion-gknn-wesley/all_kys"  # Change this path
    sims = load_cgyro_simulations(root_dir)
    print(f"Loaded {len(sims)} CGYRO simulations.")

    flux_data, tag_labels = evaluate_fluxes_from_simulations(sims, w=0.2, u='gB', n=None)

    print("\n=== Flux Summary ===")

    for ky, gamma, flux in sorted(flux_data, key=lambda x: x[0]):
        print(f"\nky = {ky:.4f}, gamma = {gamma:.4f}")
        for ispec in range(flux.shape[0]):
            for imom, tag in enumerate(tag_labels):
                print(f"  Species {ispec} - {tag}: {flux[ispec, imom]:.5e}")

    ky_sorted, flux_matrix, target_flux_matrix = compute_flux_matrix(flux_data)

    print(target_flux_matrix)
    print(target_flux_matrix.shape)

    # === HDF5 Output ===
    output_path = "ky_spectra_data.h5"
    key = "ky_spectra"

    if not os.path.exists(output_path):
        with h5py.File(output_path, "w") as f:
            f.create_dataset(key, data=target_flux_matrix[None, ...], maxshape=(None,) + target_flux_matrix.shape)
            print(f"Created new file and saved first sample to {key}.")
    else:
        with h5py.File(output_path, "a") as f:
            dset = f[key]
            dset.resize((dset.shape[0] + 1), axis=0)
            dset[-1] = target_flux_matrix
            print(f"Appended new sample to existing dataset {key}.")


if __name__ == "__main__":
    main()