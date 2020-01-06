import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as optimize

class SetsModel(object):
    """
    Insert clever documentation here
    """

    def __init__(self, state, observable, delta):

        self.alpha_z, self.kappa, self.sigma_z = state
        self.zbar = self.alpha_z/self.kappa

        self.alpha_y, self.beta, self.sigma_y = observable
        self.mu = self.alpha_y + self.beta*self.alpha_z/self.kappa

        self.sigma_z = np.asarray(self.sigma_z).reshape(2, 1)
        self.sigma_y = np.asarray(self.sigma_y).reshape(2, 1)

        self.delta = delta

        try:
            self.Sigma_inv = np.linalg.inv(np.vstack([self.sigma_y.T, self.sigma_z.T]))
        except np.linalg.LinAlgError:
            self.Sigma_inv = np.eye(2)
            self.Sigma_inv[0, 0] = 1e16
            self.Sigma_inv[1, 1] = 1/(np.vstack([self.sigma_y.T, self.sigma_z.T])[1, 1])


        self.W = self.Sigma_inv.T @ self.Sigma_inv
        self.prop = (self.W[1, 1] - self.W[0, 1]**2/self.W[0, 0])


    def stationary_z(self, worstcase=False):
        """
        Calculates the stationary distribution of z
        """

        if worstcase:
            zbar, kappa, sigma_z = self.zbar_tilde, self.kappa_tilde, self.sigma_z
        else:
            zbar, kappa, sigma_z = self.zbar, self.kappa, self.sigma_z

        var = (sigma_z.T @ sigma_z)[0][0]/(2*kappa)

        return norm(zbar, np.sqrt(var))

    def worst_case(self, tilting, theta=np.inf, discounted=False, z=None):
        """
        Given the tilting function this function calculates the worst-case models
        """
        if z is None:
            z = self.zbar

        self.xi0, self.xi1, self.xi2 = tilting
        self.discounted = discounted
        self.theta = theta

        # Calculate the elements of the value function
        self.l_star, self.s_0, self.s_1, self.u_0, self.u_1 = self.valuefunc(z=z, theta=theta)

        Sigma = np.vstack([self.sigma_y.T, self.sigma_z.T])
        eta_0 = self.s_0 + self.u_0
        eta_1 = self.s_1 + self.u_1

        dist_0 = Sigma @ eta_0
        dist_1 = Sigma @ eta_1

        # Calculate the parameters under the worst-case model
        self.beta_tilde = self.beta + dist_1[0, 0]
        self.kappa_tilde = self.kappa - dist_1[1, 0]
        self.zbar_tilde = self.zbar + dist_0[1, 0]/self.kappa_tilde

        self.alpha_z_tilde = self.kappa_tilde*self.zbar_tilde
        self.alpha_y_tilde = self.alpha_y + dist_0[0, 0] - dist_1[0, 0]*self.zbar

        self.mu_tilde = self.alpha_y_tilde + self.beta_tilde*self.zbar_tilde


    def w_func(self, l, theta, z):
        """
        Value function W for the common discount rate
        """
        if z is None:
            z = self.zbar

        mu, beta, sigma_y = self.mu, self.beta, self.sigma_y
        kappa, sigma_z = self.kappa, self.sigma_z
        mu, zbar = self.mu, self.zbar

        xi_0, xi_1, xi_2 = self.xi0, self.xi1, self.xi2
        delta = self.delta

        sigma_z2 = (sigma_z.T @ sigma_z)[0][0]
        sigma_yz = (sigma_z.T @ sigma_y)[0][0]

        ratio = l/(1 + l/theta)

        square_root = np.sqrt((delta + 2*kappa)**2 - 4*sigma_z2*xi_2*(1+l/theta))
        w2 = (delta + 2*kappa - square_root)/(2*sigma_z2)
        W2 = -ratio*w2

        if self.xi0 == 0.0 and self.xi1 == 0.0 and self.xi2 == 0.0:
            ratio_inv = 0
            w2, W2 = 0, 0
        else:
            ratio_inv = 1/ratio

        W1 = (-l*xi_1 + (.01)*beta + (.01)*w2*sigma_yz)/(delta + kappa - w2*sigma_z2)

        W0 = (-l*xi_0 + (.01)*2*mu + sigma_z2*W2 -
              (((.01)*sigma_y+sigma_z*W1).T @ ((.01)*sigma_y+sigma_z*W1))[0][0]*ratio_inv)/delta

        if np.isclose(W2, 0.0):
            W2 = 0.0

        return W0 + 2*W1*(z-zbar) + W2*(z-zbar)**2, W0, W1, W2, w2


    def valuefunc(self, z, theta):
        """
        Calculates the value functions
        """

        mu, beta, sigma_y = self.mu, self.beta, self.sigma_y
        kappa, sigma_z = self.kappa, self.sigma_z
        xi_0, xi_1, xi_2 = self.xi0, self.xi1, self.xi2
        delta = self.delta

        sigma_z2 = (sigma_z.T @ sigma_z)[0][0]
        sigma_yz = (sigma_z.T @ sigma_y)[0][0]

        if self.discounted:
            if xi_2 > 0 and theta < np.inf:
                upper_bound = theta*((delta + 2*kappa)**2/(4*sigma_z2*xi_2)-1)
                if upper_bound<0:
                    upper_bound = 10000
            else:
                upper_bound = 10000
            res = optimize.minimize_scalar(lambda l: -self.w_func(l, theta, z)[0], bounds=(1e-15, upper_bound),
                                           method = 'bounded')

            if xi_0 == 0.0 and xi_1 == 0.0 and xi_2 == 0.0:
                l_star = np.inf
                W0, W1, W2 = 0.0, 0.0, 0.0
            else:
                l_star = res.x
                W0, W1, W2, w2 = self.w_func(l_star, theta, z)[1:]

            Sigma = np.hstack([(.01)*sigma_y, sigma_z])
            s_0 = -(1/l_star)*(Sigma @ np.asarray([1, W1]).reshape(2, 1))
            s_1 = -(1/l_star)*(Sigma @ np.asarray([0, W2]).reshape(2, 1))

            u_0 = -(1/theta)*(Sigma @ np.asarray([1, W1]).reshape(2, 1))
            u_1 = -(1/theta)*(Sigma @ np.asarray([0, W2]).reshape(2, 1))


        else:
            # coeffs of R2
            rho_22 = -(kappa - np.sqrt(kappa**2 - sigma_z2*xi_2))/sigma_z2

            # coeffs of R1
            denom = np.sqrt(kappa**2 - sigma_z2*xi_2)
            rho_11 = -xi_1/(2*denom)
            rho_10 = (.01)*delta*(beta - rho_22*sigma_yz)/denom

            # coeffs of r
            r_00 = (.01)*delta*(mu - rho_11*sigma_yz) - sigma_z2*rho_10*rho_11
            r_01 = (sigma_z2*rho_22 - xi_0 - sigma_z2*rho_11**2)/2
            aux = ((.01)*delta*sigma_y + rho_10*sigma_z)
            r_0m1 = -(aux.T @ aux)[0][0]/2

            if r_01 == 0:
                l_star = np.inf
            else:
                l_star = np.sqrt(r_0m1/r_01)

            s_0 = -(1/l_star)*((.01) * delta * sigma_y + sigma_z * rho_10) - sigma_z * rho_11
            s_1 = - sigma_z * rho_22

            sigma_ys = (sigma_y.T @ s_1)[0][0]
            sigma_zs = (sigma_z.T @ s_1)[0][0]

            V1 = (.01)*(beta + sigma_ys)/(delta + kappa - sigma_zs)

            u_0 = (-1/theta)*np.hstack([sigma_y, sigma_z]) @ np.asarray([1, V1]).reshape(2, 1)
            u_1 = np.zeros((2, 1))

        return l_star, s_0, s_1, u_0, u_1

    def minmax_value(self, tilting, theta=np.inf, z=None):
        """
        Calculates the minimum and maxumum value for a given xi
        """
        if z is None:
            z = self.zbar

        self.xi0, self.xi1, self.xi2 = tilting

        mu, beta, sigma_y = self.mu, self.beta, self.sigma_y
        kappa, sigma_z = self.kappa, self.sigma_z
        delta = self.delta
        zbar = self.zbar

        sigma_z2 = (sigma_z.T @ sigma_z)[0][0]
        sigma_yz = (sigma_z.T @ sigma_y)[0][0]

        if self.discounted:
            if self.xi2 > 0 and theta < np.inf:
                upper_bound = theta*((delta + 2*kappa)**2/(4*sigma_z2*self.xi2)-1)
                if upper_bound<0:
                    upper_bound = 10000
            else:
                upper_bound = 10000

            res_min = optimize.minimize_scalar(lambda l: -self.w_func(l, theta, z)[0], bounds=(1e-15, upper_bound),
                                           method = 'bounded')
            res_max = optimize.minimize_scalar(lambda l: self.w_func(l, theta, z)[0], bounds=(-upper_bound, 1e-15),
                                           method = 'bounded')

            if self.xi0 == 0.0 and self.xi1 == 0.0 and self.xi2 == 0.0:
                l_star = 0
                W0, W1, W2, w2 = self.w_func(l_star, theta, z)[1:]
                v_max = (.5)*(W0 + 2*W1*(z-zbar))
                v_min = v_max

            else:
                l_star_min = res_min.x
                W0, W1, W2, w2 = self.w_func(l_star_min, theta, z)[1:]
                v_min = (.5)*(W0 + 2*W1*(z-zbar) + W2*(z-zbar)**2)

                l_star_max = res_max.x
                W0, W1, W2, w2 = self.w_func(l_star_max, theta, z)[1:]
                v_max = (.5)*(W0 + 2*W1*(z-zbar) + W2*(z-zbar)**2)


        return v_min, v_max


    def chernoff_objfunc(self, r, theta=np.inf, z=None):
        """
        For a given r and affine drift distortion (worstcase method must be executed before),
        this method calculates -psi (eignrvalue of the generator) - the min is chernoff entropy
        """

        # Pull out useful info
        alpha_z, kappa, sigma_z = self.alpha_z, self.kappa, self.sigma_z
        s2 = (sigma_z.T @ sigma_z)[0][0]
        l_star, s_0, s_1, u_0, u_1 = self.valuefunc(z=z, theta=theta)

        eta_0 = s_0 + u_0
        eta_1 = s_1 + u_1

        kappa_tilde = kappa - r*(sigma_z.T @ eta_1)[0][0]
        alpha_z_tilde = alpha_z + r*(sigma_z.T @ eta_0)[0][0]

        zeta_0 = -r*(r-1)*(eta_0.T @ eta_0)[0][0]
        zeta_1 = -r*(r-1)*(eta_0.T @ eta_1)[0][0]
        zeta_2 = -r*(r-1)*(eta_1.T @ eta_1)[0][0]

        lambda_2 = (kappa_tilde - np.sqrt(kappa_tilde**2 + zeta_2*s2))/s2
        lambda_1 = - (zeta_1 - alpha_z_tilde*lambda_2)/np.sqrt(kappa_tilde**2 + zeta_2*s2)
        psi = .5*zeta_0 - lambda_1*alpha_z_tilde - .5*(lambda_2 + lambda_1**2)*s2

        return -psi


    def chernoff(self, theta=np.inf, z=None):

        res = optimize.minimize_scalar(lambda r: self.chernoff_objfunc(r, theta, z=z),
                                          bounds = (0, 1), method = 'bounded')

        if res.fun == 0:
            HL = np.inf
        else:
            HL = np.log(2) / (-res.fun)

        # half-life and Chernoff entropy
        return HL, -res.fun


    def __calibr(self, tilting, theta, discounted, z=None):
        """
        Auxiliary function for the half-life calibration
        """
        self.worst_case(tilting, theta=theta, discounted=discounted, z=z)
        hl = self.chernoff(theta=theta, z=z)[0]

        return hl


    def target_HL(self, HL, kappa_bar, infinite_theta=True, discounted=False, z=None):

        if infinite_theta and kappa_bar is None:
            # original sets paper's calibration (Example 1)
            xi_2 = optimize.bisect(lambda xi2: HL - self.__calibr([0.0, 0.0, xi2], np.inf, discounted, z=z),
                                   0.001, .6)
            return [0.0, 0.0, xi_2], np.inf

        elif infinite_theta and kappa_bar is not None:
            # original sets paper's calibration (Example 2)
            xi_2 = self.generate_xi(kappa_bar)[2]
            xi_0 = optimize.bisect(lambda xi0: HL-self.__calibr([xi0, 0.0, xi_2], np.inf, discounted, z=z),
                                   0.00001, 100.0)
            return [xi_0, 0.0, xi_2], np.inf

        else:
            # MAIN METHOD: calibrate theta to match HL for given kappa_bar
            tilting = self.generate_xi(kappa_bar)
            theta = optimize.bisect(lambda th: HL - self.__calibr(tilting, th, discounted, z=z), 1e-5, 1e5)

            return tilting, theta


    def relative_entropy(self):
        """
        For a given kappa_bar, this method calculates the worst-case drift distrortion
        and the associated relative entropy
        """

        # Pull out useful info
        sigma_z = self.sigma_z
        s2 = (sigma_z.T @ sigma_z)[0][0]

        eta_0 = self.s_0 + self.u_0
        eta_1 = self.s_1 + self.u_1
        kappa_tilde = self.kappa_tilde
        alpha_z_tilde = self.alpha_z_tilde

        rho_2 = (eta_1.T @ eta_1)[0][0]/2/kappa_tilde
        rho_1 = (eta_0.T @ eta_1)[0][0]/kappa_tilde + rho_2*alpha_z_tilde/kappa_tilde
        re = (eta_0.T @ eta_0)[0][0]/2 + rho_1*alpha_z_tilde + rho_2*s2/2

        return re


    def generate_xi(self, kappa_bar):
        """
        Calculates the coeffs of the tiliting function from kappa_bar
        """
        kappa = self.kappa
        #xi_2 = self.prop*(kappa-kappa_bar)**2
        #xi_2 = (kappa-kappa_bar)**2 / (self.sigma_z.T @ self.sigma_z)[0][0]
        xi_2 = ((kappa-kappa_bar)/self.sigma_z[1])**2 

        return [0.0, 0.0, xi_2]

    def table(self, targets, infinite_theta=True, discounted=False, z=None):
        """
        Creates the table for different targets, where targets = (kappa_bar, HL):
            (1) if HL is None, not targeting half-life
            (2) if HL is given and
                - infinite_theta is True -> calibrate HL with xi_0
                      unless kappa_bar is None -> calibrate HL with kappa_bar
                - infinite_theta is False  -> calibrate HL with theta

        """

        print("    HL   |  kappa_bar  |   alpha_y   |   beta   |  alpha_z   |   kappa   |   z_bar   |   \\theta  |  RE ")
        print("-------------------------------------------------------------------------------------------------------")
        print("&$\infty$ &  {: 0.5f}    &   {: 0.5f}    & {: 0.5f}   &  {: 0.5f}    &  {: 0.5f}   &  {: 0.5f}   & $\infty$  & {: 0.5f} \\\\".format(self.kappa, self.alpha_y, self.beta, self.alpha_z, self.kappa, self.zbar, 0))

        for kappa_bar, HL in targets:
            if infinite_theta:
                if HL is None:
                    # Statistician's or first stage discounted problem
                    tilting = self.generate_xi(kappa_bar)
                    self.worst_case(tilting, theta=np.inf, discounted=discounted, z=z)

                elif kappa_bar is None:
                    # Original sets paper, example 1: set xi_0=0, calibrate kappa_bar
                    tilting = self.target_HL(HL, kappa_bar, infinite_theta=True,
                                         discounted=discounted, z=z)[0]
                    self.worst_case(tilting, theta=np.inf, discounted=discounted, z=z)
                    kappa_bar = self.kappa - np.sqrt(tilting[2]/self.prop)

                else:
                    tilting = self.target_HL(HL, kappa_bar, infinite_theta=True,
                                         discounted=discounted, z=z)[0]
                    self.worst_case(tilting, theta=np.inf, discounted=discounted, z=z)

                self.print_out_params(kappa_bar)

            else:
                tilting, theta = self.target_HL(HL, kappa_bar, infinite_theta=False,
                                         discounted=discounted, z=z)
                self.worst_case(tilting, theta=theta, discounted=discounted, z=z)

                self.print_out_params(kappa_bar)



    def print_out_params(self, kappa_bar):
        '''
        Print out the parameters and other objects of interests for
        '''
        alpha_y, beta = self.alpha_y_tilde, self.beta_tilde
        alpha_z, kappa = self.alpha_z_tilde, self.kappa_tilde
        mu = self.mu
        mu_tilde, zbar = self.mu_tilde, self.zbar_tilde
        dc_gap = mu_tilde-mu

        HL = self.chernoff(theta=self.theta)[0]
        RE = self.relative_entropy()

        q = np.sqrt(2*RE)

        print("&  {:5.0f}  &  {: 1.5f}    &   {: 1.5f}    & {: 1.5f}".format(HL, kappa_bar, alpha_y, beta) +
              "   &  {: 1.5f}    &  {: 1.5f}".format(alpha_z, kappa) +
              "   &  {: 1.3f}   &  {:1.3f}    &  {:1.3f}".format(dc_gap, self.xi0, RE) + "\\\\")




#==============================================================
# Other functions necessary to generate the figrues
#==============================================================

# Parameter values come from the first specification of the robust planner problem
phi_bar = .499
beta_bar = 1

alpha_hat = 0
kappa_hat = .169
sigma = .195
delta = .01
x0 =  alpha_hat/kappa_hat


def tilted_entropy(alpha, kappa, xi_0, xi_1, xi_2,
                   alpha_hat=alpha_hat, kappa_hat=kappa_hat, sigma=sigma, delta=delta, x0=x0):
    rho_0 = ((alpha-alpha_hat)/sigma)**2 - xi_0
    rho_1 = ((kappa_hat-kappa)/sigma)*((alpha-alpha_hat)/sigma) - xi_1
    rho_2 = ((kappa_hat-kappa)/sigma)**2 - xi_2

    xbar = alpha_hat/kappa_hat
    mean = alpha/kappa
    var = sigma**2/(2*kappa)

    varrho = rho_0/2 + rho_1*(mean-xbar) + (.5)*rho_2*(var + (mean-xbar)**2) + \
            (delta*(rho_2*(mean-xbar) + rho_1)/(delta+kappa))*(x0-mean) + \
            (delta*rho_2/(2*(delta+2*kappa)))*((x0-mean)**2 - var)

    return varrho/delta

def entropy(alpha, kappa, alpha_hat=alpha_hat, kappa_hat=kappa_hat, sigma=sigma, delta=delta, x0=x0):
    rho_0 = ((alpha-alpha_hat)/sigma)**2
    rho_1 = ((kappa_hat-kappa)/sigma)*((alpha-alpha_hat)/sigma)
    rho_2 = ((kappa_hat-kappa)/sigma)**2

    xbar = alpha_hat/kappa_hat
    mean = alpha/kappa
    var = sigma**2/(2*kappa)

    varrho = rho_0/2 + rho_1*(mean-xbar) + (.5)*rho_2*(var + (mean-xbar)**2) + \
            (delta*(rho_2*(mean-xbar) + rho_1)/(delta+kappa))*(x0-mean) + \
            (delta*rho_2/(2*(delta+2*kappa)))*((x0-mean)**2 - var)

    return varrho/delta

def isovalue(alpha, kappa, phi_bar=phi_bar, beta_bar=beta_bar, delta=delta, x0=x0):
    v0 = (.01)*(phi_bar/delta + alpha*beta_bar/delta/(delta + kappa))
    v1 = (.01)*beta_bar/(delta + kappa)

    return v0 + v1*x0



def xi_generator(alpha, kappa, alpha_hat=alpha_hat, kappa_hat=kappa_hat, sigma=sigma):
    eta_0 = (alpha-alpha_hat)/sigma
    eta_1 = (kappa_hat-kappa)/sigma

    return eta_0*eta_0, eta_0*eta_1, eta_1*eta_1


def instant_pairs(xi_0, xi_1, xi_2, alpha_hat=alpha_hat, kappa_hat=kappa_hat, sigma=sigma):

    if xi_1<0:
        kappa1 = kappa_hat - np.sqrt(xi_2)*sigma
        alpha1 = alpha_hat - np.sqrt(xi_0)*sigma

        kappa2 = kappa_hat + np.sqrt(xi_2)*sigma
        alpha2 = alpha_hat + np.sqrt(xi_0)*sigma

    else:
        kappa1 = kappa_hat + np.sqrt(xi_2)*sigma
        alpha1 = alpha_hat - np.sqrt(xi_0)*sigma

        kappa2 = kappa_hat - np.sqrt(xi_2)*sigma
        alpha2 = alpha_hat + np.sqrt(xi_0)*sigma

    return [kappa1, kappa2], [alpha1, alpha2]


def instant_constraint(alpha, kappa, xi_0, xi_1, xi_2, alpha_hat=alpha_hat, kappa_hat=kappa_hat, sigma=sigma):

    rho_0 = ((alpha-alpha_hat)/sigma)**2 - xi_0
    rho_1 = ((kappa_hat-kappa)/sigma)*((alpha-alpha_hat)/sigma) - xi_1
    rho_2 = ((kappa_hat-kappa)/sigma)**2 - xi_2

    if type(rho_2)==float:
        return_matrix = rho_1**2 - rho_0*rho_2
        if rho_2>0 or rho_0>0 or rho_1**2 - rho_0*rho_2>0:
            return 1000
        return 0

    else:
        return_matrix = np.zeros_like(rho_2)
        return_matrix[rho_2>0] = 1000
        return_matrix[rho_0>0] = 1000
        return_matrix[rho_1**2 - rho_0*rho_2>0] = 1000

        return return_matrix



def drift_distortions_new(sets, kappa_bar, halflife, z_grid):

    ZZ = z_grid.reshape(1, len(z_grid))               # this grid is for the drift (fixed l*)

    # define the benchmark drift
    mu, beta, sigma_y, kappa, sigma_z = sets.mu, sets.beta, sets.sigma_y, sets.kappa, sets.sigma_z
    drift_Y = mu + beta*ZZ
    drift_Z = -kappa*ZZ

    # number of initial z where we want to evaluate l*(z) and use xi_2 and theta calibrated above
    initial_z = [sets.zbar]
    nn = len(initial_z)

    # define storing matrices for the discounted and undiscounted case
    h0_fin_theta, h1_fin_theta = np.zeros((2, nn)), np.zeros((2, nn))
    s0_fin_theta, s1_fin_theta = np.zeros((2, nn)), np.zeros((2, nn))
    u0_fin_theta, u1_fin_theta = np.zeros((2, nn)), np.zeros((2, nn))


    k_b, hl = kappa_bar, halflife
    tilting, theta = sets.target_HL(hl, k_b, infinite_theta=True, discounted=True)

    for i, zz in enumerate(initial_z):
        sets.worst_case(tilting, theta=theta, discounted=True, z=zz)
        h0_fin_theta[:, i]= sets.s_0.squeeze()
        h1_fin_theta[:, i]= sets.s_1.squeeze()


    for i, zz in enumerate(initial_z):
        sets.worst_case([0] + tilting[1:], theta=theta, discounted=True, z=zz)
        s0_fin_theta[:, i]= sets.s_0.squeeze()
        s1_fin_theta[:, i]= sets.s_1.squeeze()

    u0_fin_theta = h0_fin_theta - s0_fin_theta
    u1_fin_theta = h1_fin_theta - s1_fin_theta

    common_disc_rate_game = [s0_fin_theta, s1_fin_theta, u0_fin_theta, u1_fin_theta]

    # Discounted case with average initial z
    distZ_s = sigma_z.T @ (s0_fin_theta + s1_fin_theta*ZZ)
    distZ_u = sigma_z.T @ (u0_fin_theta + u1_fin_theta*ZZ)
    distZ = distZ_s + distZ_u

    # Discounted case with average initial z
    distY_s = sigma_y.T @ (s0_fin_theta + s1_fin_theta*ZZ)
    distY_u = sigma_y.T @ (u0_fin_theta + u1_fin_theta*ZZ)
    distY = distY_s + distY_u


    return drift_Z, distZ_s, distZ_u, distZ, drift_Y, distY_s, distY_u, distY, common_disc_rate_game




def drift_distortions(sets, kappa_bar, halflife, z_grid, infinite_theta=False):
    k_b, hl = kappa_bar, halflife
    tilting, theta = sets.target_HL(hl, k_b, infinite_theta=infinite_theta, discounted=True)
    tilting, theta_und = sets.target_HL(hl, k_b, infinite_theta=infinite_theta, discounted=False)

    ZZ = z_grid.reshape(1, len(z_grid))               # this grid is for the drift (fixed l*)

    # define the benchmark drift
    mu, beta, sigma_y, kappa, sigma_z = sets.mu, sets.beta, sets.sigma_y, sets.kappa, sets.sigma_z
    drift_Y = mu + beta*ZZ
    drift_Z = -kappa*ZZ

    # number of initial z where we want to evaluate l*(z) and use xi_2 and theta calibrated above
    initial_z = [sets.zbar]
    nn = len(initial_z)

    # define storing matrices for the discounted and undiscounted case
    s0_fin_theta, s1_fin_theta = np.zeros((2, nn)), np.zeros((2, nn))
    u0_fin_theta, u1_fin_theta = np.zeros((2, nn)), np.zeros((2, nn))
    s0_fin_theta_und, s1_fin_theta_und = np.zeros((2, nn)), np.zeros((2, nn))
    u0_fin_theta_und, u1_fin_theta_und = np.zeros((2, nn)), np.zeros((2, nn))

    for i, zz in enumerate(initial_z):
        sets.worst_case(tilting, theta=theta, discounted=True, z=zz)
        s0_fin_theta[:, i]= sets.s_0.squeeze()
        u0_fin_theta[:, i]= sets.u_0.squeeze()
        s1_fin_theta[:, i]= sets.s_1.squeeze()
        u1_fin_theta[:, i]= sets.u_1.squeeze()

        sets.worst_case(tilting, theta=theta_und, discounted=False, z=zz)
        s0_fin_theta_und[:, i]= sets.s_0.squeeze()
        u0_fin_theta_und[:, i]= sets.u_0.squeeze()
        s1_fin_theta_und[:, i]= sets.s_1.squeeze()
        u1_fin_theta_und[:, i]= sets.u_1.squeeze()

    statistician_game = [s0_fin_theta_und, s1_fin_theta_und, u0_fin_theta_und, u1_fin_theta_und]
    common_disc_rate_game = [s0_fin_theta, s1_fin_theta, u0_fin_theta, u1_fin_theta]

    # Discounted case with average initial z
    distZ_s = sigma_z.T @ (s0_fin_theta + s1_fin_theta*ZZ)
    distZ_u = sigma_z.T @ (u0_fin_theta + u1_fin_theta*ZZ)
    distZ = distZ_s + distZ_u

    # Discounted case with average initial z
    distY_s = sigma_y.T @ (s0_fin_theta + s1_fin_theta*ZZ)
    distY_u = sigma_y.T @ (u0_fin_theta + u1_fin_theta*ZZ)
    distY = distY_s + distY_u

    # Undiscounted case with average initial z
    distZ_s_und = sigma_z.T @ (s0_fin_theta_und + s1_fin_theta_und*ZZ)
    distZ_u_und = sigma_z.T @ (u0_fin_theta_und + u1_fin_theta_und*ZZ)
    distZ_und = distZ_s_und + distZ_u_und



    return drift_Z, distZ_s_und, distZ_u_und, distZ_und, \
                    distZ_s, distZ_u, distZ, drift_Y, distY_s, distY_u, distY, statistician_game, common_disc_rate_game



def NBER_Shade(ax, start_date):
    """
    This function adds NBER recession bands to a Matplotlib Figure object.
    ax         : axis
    start_date : start date for the sample, form: yyyy-mm-dd
    """

    # load the NBER recession dates
    NBER_Dates = pd.read_csv('NBER_dates.txt')
    sample_1 = pd.Timestamp(start_date) <= pd.DatetimeIndex(NBER_Dates['Peak'])
    sample_2 = pd.Timestamp(start_date) <= pd.DatetimeIndex(NBER_Dates['Trough'])
    NBER_Dates = NBER_Dates[sample_1 + sample_2]

    # for loop generates recession bands!
    for i in NBER_Dates.index:
        ax.axvspan(NBER_Dates['Peak'][i], NBER_Dates['Trough'][i], facecolor='grey', alpha=0.15)
