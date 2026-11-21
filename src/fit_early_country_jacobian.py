"""Same estimator with the actual earlier loop horizons, for two frozen concepts."""
import fit_country_jacobian as fit

if __name__=='__main__':
    fit.SITES[:]=[11,23]
    fit.COUNTRIES[:]=['Germany','China']
    fit.main()
