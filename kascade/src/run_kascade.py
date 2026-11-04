from buildclass import kascade
from buildbasis import add_basis_to_kascade
from setup_inversion import generate_inversion_components
from optimizer_functions import drive_fistasolver, get_lambda_for_nonlin, solve_log10_adam

if __name__=='__main__':
    # file containing the data
    filepath = '/Users/sbdas/Documents/Research/SpaceWeather/Turbulence/kascade/datafiles/Jul162026_psd_arr_enc22_4srijan.sav'
    # Loading the JSON file with key mapping
    config_path = 'config.json'

    # creating an instance of the kascade class
    ksc = kascade(filepath, config_path)

    # now adding on the basis functions to the kascade class instance
    add_basis_to_kascade(ksc, config_path)

    # generating the kernel for inversion
    generate_inversion_components(ksc)

    # obtaining the first estimates from FISTA inversion
    drive_fistasolver(ksc, plot=True)

    # obtaining the lambda for non-linear inversion
    get_lambda_for_nonlin(ksc)

    # adam non linear solver
    solve_log10_adam(ksc)


