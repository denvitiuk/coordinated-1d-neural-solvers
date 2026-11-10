"""Запускает все проверки: python checks/run_checks.py"""

from run_laplace import run_laplace
from test_cgls import test_cgls
from test_descent_exact_step import test_descent_exact_step
from test_grad_adjoint import test_grad_adjoint
from test_plate import test_plate
from test_eco import test_eco
from test_eco_inverse import test_eco_inverse
from test_multi import test_plate_as_two_subsystems, test_random_three, test_warsaw_kyiv
from test_krylov import test_cr_symmetric, test_factor_matches_thomas, test_gmres_general
from test_transpose import test_transpose
from test_pcr import test_pcr

if __name__ == "__main__":
    test_transpose()
    test_factor_matches_thomas()
    test_pcr()
    test_grad_adjoint()
    test_descent_exact_step()
    test_cgls()
    test_gmres_general()
    test_cr_symmetric()
    run_laplace()
    test_plate()
    test_random_three()
    test_plate_as_two_subsystems()
    test_warsaw_kyiv()
    test_eco()
    test_eco_inverse()
    print("ALL CHECKS PASSED")
