import json
import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt; plt.ion()

class kascade:
    def __init__(self, filepath, config_path):
        # reading teh data from the .sav file
        data = readsav(filepath, python_dict=True)

        # reading the config file defined by the user
        with open(config_path, 'r') as f:
            config = json.load(f)

        # storing the config attributes in the kascade class instance
        keymappings = config.get("keymappings", {})
        for attr_name, sav_key in keymappings.items():
            if sav_key in data:
                setattr(self, attr_name, data[sav_key])
            else:
                raise KeyError(f"Key '{sav_key}' not found in the .sav file.")
        
        userinit_params = config.get("userinit_params", {})
        for param_name, param_value in userinit_params.items():
            if(param_value == -1): continue
            else: setattr(self, param_name, param_value)

        # finding the indices of the frequency range
        self.fmin_idx = np.argmin(np.abs(self.F - self.fmin))
        self.fmax_idx = np.argmin(np.abs(self.F - self.fmax))

        # clipping to frequency range and assiging the total number of freqs
        self.F = self.F[self.fmin_idx:self.fmax_idx+1]
        self.nfreq = len(self.F)

        # removing the NaN entries from the data arrays
        finite_idx = np.isfinite(self.VSW)

        # removing all the NaN entries from the data arrays
        self.VSW   = self.VSW[finite_idx][::self.tskip]
        self.VA    = self.VA[finite_idx][::self.tskip]
        self.TH_BV = self.TH_BV[finite_idx][::self.tskip]

        # converting to meters from km
        self.VSW = self.VSW * 1e3
        self.VA = self.VA * 1e3
        self.rhoi = self.rhoi * 1e3
        self.di = self.di * 1e3

        # adjusting the TH_BV convention and converting to radians
        self.TH_BV = np.radians(np.abs(np.abs(self.TH_BV-90)-90))

        # initiating the kperp and kpara arrays
        self.rhoi, self.di = np.nanmean(self.rhoi), np.nanmean(self.di)
        self.kperp = np.logspace(self.logkperp_min, self.logkperp_max, self.nkperp) / self.rhoi
        self.kpara = np.logspace(self.logkpara_min, self.logkpara_max, self.nkpara) / self.di

        # we need to check this for synthetic data since the P_theta_f is not present in the synthetic data
        if "P_theta_f" in keymappings:
            # slicing the P_theta_f data 
            self.P_theta_f = self.P_theta_f[finite_idx, :][::self.tskip, self.fmin_idx:self.fmax_idx+1]
            # flattening since that is how we want the data
            self.d = np.reshape(self.P_theta_f, (-1), 'C')
        else:
            self.P_theta_f = None
            self.d = None


if __name__=='__main__':
    # file containing the data
    filepath = '/Users/sbdas/Documents/Research/SpaceWeather/Turbulence/kascade/datafiles/Jul162026_psd_arr_enc22_4srijan.sav'
    # Loading the JSON file with key mapping
    config_path = 'config.json'

    # creating an instance of the kascade class
    ksc = kascade(filepath, config_path)

    # making a plot of the loaded data for sanity check
    plt.figure()
    plt.loglog(ksc.F, ksc.P_theta_f[2000], '.k', label='original data')
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD [$\mathrm{nT}^2$/Hz]')