import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import seaborn as sns
import os
import pickle
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

colors = sns.color_palette()


def confidence_band(joint_dens, perc):
    return optimize.brentq(lambda x : np.sum(joint_dens[joint_dens < x]) - perc, 1e-15, .8)

def half_split(x,y, bar = None):
    if bar == None:
        ybar = np.mean(y)
        up_ind = np.where(y >= ybar)
        down_ind = np.where(y < ybar)

        y_up = y[up_ind]
        x_up = x[up_ind]
        sort_ind = np.argsort(x_up)
        x_up = x_up[sort_ind]
        y_up = y_up[sort_ind]

        y_down = y[down_ind]
        x_down = x[down_ind]
        sort_ind = np.argsort(x_down)
        x_down = x_down[sort_ind]
        y_down = y_down[sort_ind]
    else:
        ybar = bar
        up_ind = np.where(y >= ybar)
        down_ind = np.where(y < ybar)

        y_up = y[up_ind]
        x_up = x[up_ind]
        sort_ind = np.argsort(x_up)
        x_up = x_up[sort_ind]
        y_up = y_up[sort_ind]

        y_down = y[down_ind]
        x_down = x[down_ind]
        sort_ind = np.argsort(x_down)
        x_down = x_down[sort_ind]
        y_down = y_down[sort_ind]
    
    return x_down, y_down, x_up, y_up


class Model(object):

    def __init__(self, model):

        #=================================================#
        # Used parameters
        #=================================================#
        self.delta = model["delta"]

        # Single capital economy
        self.alpha_c_hat, self.beta_hat = model["alpha_c_hat"], model["beta_hat"]
        self.alpha_z_hat, self.kappa_hat = model["alpha_z_hat"], model["kappa_hat"]
        self.sigma_c, self.sigma_z_1cap = model["sigma_c"], model["sigma_z_1cap"]
        self.zbar, self.cons_1cap = model["zbar"], model["cons_1cap"]
        self.stdev_z_1cap = model["stdev_z_1cap"]
        self.H0, self.H1 = model["H0"], model["H1"]

        # Two capital stocks
        self.alpha_k1_hat, self.alpha_k2_hat = model["alpha_k1_hat"], model["alpha_k2_hat"]
        self.beta1_hat, self.beta2_hat = model["beta1_hat"], model["beta2_hat"]
        self.sigma_k1, self.sigma_k2 = model["sigma_k1"], model["sigma_k2"]
        self.sigma_z = model["sigma_z"]
        self.A1, self.A2, self.phi1, self.phi2 = model["A1"], model["A2"], model["phi1"], model["phi2"]

        # Worrisome model
        self.alpha_z_tilde, self.kappa_tilde = model["alpha_z_tilde"], model["kappa_tilde"]

        self.V_noR, self.V = model["V_noR"], model["V"]
        self.q = model["q"]

        self.A_1cap = model["A_1cap"]
        self.phi_1cap = model["phi_1cap"]
        self.alpha_k_hat = model["alpha_k_hat"]
        self.consumption_investment = model["consumption_investment"]
        self.investment_capital = model["investment_capital"]

        #=================================================#
        # Grid parameters
        #=================================================#
        self.I, self.J = model["I"], model["J"]
        self.pii, self.rr = model["pii"], model["rr"]
        self.zz = model["zz"]
        self.dz, self.dr = model["dz"], model["dr"]
        self.T, self.inner = model["T"], model["inner"]

        #=================================================#
        # Optimal decisions
        #=================================================#
        self.d1_noR, self.d2_noR = model["d1_noR"], model["d2_noR"]
        self.d1, self.d2 = model["d1"], model["d2"]

        # worst-case means shifts of Brownian shocks
        self.h1_dist, self.h2_dist = model["h1_dist"], model["h2_dist"]
        self.hz_dist = model["hz_dist"]

        # local uncertainty prices (minus h_dist)
        self.h1, self.h2, self.hz = model["h1"], model["h2"], model["hz"]

        #=================================================#
        # Drifts
        #=================================================#
        self.mu_1_noR, self.mu_r_noR, self.mu_z_noR = model["mu_1_noR"], model["mu_r_noR"], model["mu_z_noR"]
        self.mu_1, self.mu_r, self.mu_z = model["mu_1"], model["mu_r"], model["mu_z"]
        self.mu_1_wc, self.mu_r_wc, self.mu_z_wc = model["mu_1_wc"], model["mu_r_wc"], model["mu_z_wc"]
        self.drift_of_pii()

        #=================================================#
        # Distributions
        #=================================================#
        self.g_noR_dist, self.g_noR = model["g_noR_dist"], model["g_noR"]
        self.g_dist, self.g = model["g_dist"], model["g"]
        self.g_wc_dist, self.g_wc = model["g_wc_dist"], model["g_wc"]
        self.g_wc_noR_dist, self.g_wc_noR = model["g_wc_noR_dist"], model["g_wc_noR"]

        self.g_noR[self.g_noR<0] = 0.0
        self.g[self.g<0] = 0.0
        self.g_wc[self.g_wc<0] = 0.0

        # Normalize joint distribution for variables pii and zz (w.r.t. Lebesque measure)
        self.f_noR_dist = self.g_noR_dist[1:, :] * (self.pii[1:, :] - self.pii[:-1, :]) * self.dz
        self.f_dist = self.g_dist[1:, :] * (self.pii[1:, :] - self.pii[:-1, :]) * self.dz
        self.f_wc_dist = self.g_wc_dist[1:, :] * (self.pii[1:, :] - self.pii[:-1, :]) * self.dz

        self.f_noR_dist = self.f_noR_dist/self.f_noR_dist.sum()
        self.f_dist = self.f_dist/self.f_dist.sum()
        self.f_wc_dist = self.f_wc_dist/self.f_wc_dist.sum()

        #=================================================#
        # Consumption
        #=================================================#
        self.cons_noR, self.cons = model["cons_noR"], model["cons"]
        self.cons_noR_vec, self.cons_noR_density = model["cons_noR_vec"], model["cons_noR_density"]
        self.cons_vec, self.cons_density = model["cons_vec"], model["cons_density"]
        self.cons_wc_vec, self.cons_wc_density = model["cons_wc_vec"], model["cons_wc_density"]

        self.logC_mu_noR, self.logC_sigma_noR = model["logC_mu_noR"], model["logC_sigma_noR"]
        self.logC_mu, self.logC_sigma = model["logC_mu"], model["logC_sigma"]
        self.logC_mu_wc, self.logC_sigma_wc = model["logC_mu_wc"], model["logC_sigma_wc"]

        #=================================================#
        # Uncertainty prices
        #=================================================#
        self.h12_vec, self.h12_density = model["h12_vec"], model["h12_density"]
        self.hz_vec, self.hz_density = model["hz_vec"], model["hz_density"]
        #self.rf_vec, self.rf_density = model["rf_vec"], model["rf_density"]
        #self.riskfree = model["riskfree"]
        self.shock_price_12 = model["shock_price_12"]
        self.shock_price_z = model["shock_price_z"]

        #=================================================#
        # Impulse Response Functions
        #=================================================#
        self.R_irf, self.Z_irf = model["R_irf"], model["Z_irf"]

        z_baseline = norm(loc= self.zbar, scale = self.stdev_z_1cap)

        # Find z indeces for the .1 and .9 deciles (under the baseline)
        self.ind_ld_z = np.argmin((self.zz[0, :] - z_baseline.ppf(.1))**2)
        self.ind_med_z = np.argmin((self.zz[0, :] - z_baseline.ppf(.5))**2)
        self.ind_ud_z = np.argmin((self.zz[0, :] - z_baseline.ppf(.9))**2)

        mycdf = self.f_dist.sum(1).cumsum()
        for j in range(self.I-1):
            if mycdf[j] > .5:
                self.ind_med_r = j + 1
                break

        for j in range(self.I-1):
            if mycdf[j] > .1:
                self.ind_ld_r = j
                break

        for j in range(self.I-1):
            if mycdf[j] > .9:
                self.ind_ud_r = j
                break

        mycdf_noR = self.f_noR_dist.sum(1).cumsum()
        for j in range(self.I-1):
            if mycdf_noR[j] > .5:
                self.ind_med_r_noR = j + 1
                break

        for j in range(self.I-1):
            if mycdf_noR[j] > .1:
                self.ind_ld_r_noR = j
                break

        for j in range(self.I-1):
            if mycdf_noR[j] > .9:
                self.ind_ud_r_noR = j
                break


    def drift_of_pii(self):

        sigma_k1, sigma_k2, pii, zz = self.sigma_k1, self.sigma_k2, self.pii, self.zz
        h1_dist, h2_dist, hz_dist = self.h1_dist, self.h2_dist, self.hz_dist
        d1_noR, d2_noR, phi1, phi2 = self.d1_noR, self.d2_noR, self.phi1, self.phi2
        alpha_k1_hat, alpha_k2_hat = self.alpha_k1_hat, self.alpha_k2_hat
        beta1_hat, beta2_hat = self.beta1_hat, self.beta2_hat
        d1, d2 = self.d1, self.d2

        jensen_R = (.01)**2*((sigma_k1 @ sigma_k1)*(1-pii) -
                             (sigma_k2 @ sigma_k2)*pii +
                             (sigma_k1 @ sigma_k2)*(2*pii - 1))

        wc_adj_R = (.01)*pii*(1-pii)*((sigma_k2-sigma_k1)[0]*h1_dist +
                                      (sigma_k2-sigma_k1)[1]*h2_dist +
                                      (sigma_k2-sigma_k1)[2]*hz_dist)

        self.mu_k1_noR = d1_noR - d1_noR**2 * phi1/2 + (.01)*(alpha_k1_hat + beta1_hat*zz)
        self.mu_k2_noR = d2_noR - d2_noR**2 * phi2/2 + (.01)*(alpha_k2_hat + beta2_hat*zz)
        self.mu_k1 = d1 - d1**2 * phi1/2 + (.01)*(alpha_k1_hat + beta1_hat*zz)
        self.mu_k2 = d2 - d2**2 * phi2/2 + (.01)*(alpha_k2_hat + beta2_hat*zz)

        self.mu_pii_noR = 100*(pii * (1-pii) * (self.mu_k2_noR - self.mu_k1_noR + jensen_R))
        self.mu_pii = 100*(pii*(1-pii) * (self.mu_k2 - self.mu_k1 + jensen_R))
        self.mu_pii_wc = 100*(pii*(1-pii)*(self.mu_k2 - self.mu_k1 + jensen_R) + wc_adj_R)



    def figure_robustcontrol(self, ax1_t, cut=25, left_top_ylim=None):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_med_r = self.ind_med_r
        pii, zz = self.pii, self.zz
        d1_noR, d2_noR, d1, d2 = self.d1_noR, self.d2_noR, self.d1, self.d2

        pii_cut = pii[cut:-cut, 0]

        ax1_t.plot(pii_cut, d1_noR[cut:-cut, ind_med_z], lw=3, color='k', label='without robustness')
        ax1_t.plot(pii_cut, d1_noR[cut:-cut, ind_ud_z], lw=1, color='k', linestyle=':', alpha=.9)
        ax1_t.fill_between(pii_cut, d1_noR[cut:-cut, ind_ld_z], d1_noR[cut:-cut, ind_ud_z],
                           lw=3, color='k', alpha=.1)
        ax1_t.plot(pii_cut, d1[cut:-cut, ind_med_z], lw=3, color=colors[3], label='robust controls')
        ax1_t.plot(pii_cut, d1[cut:-cut, ind_ud_z], lw=1, color=colors[3], linestyle=':')
        ax1_t.fill_between(pii_cut, d1[cut:-cut, ind_ld_z], d1[cut:-cut, ind_ud_z],
                           lw=3, color=colors[3], alpha=.2)

        ax1_t.plot(pii_cut, d2_noR[cut:-cut, ind_med_z], lw=3, linestyle='--', color='k')
        ax1_t.plot(pii_cut, d2_noR[cut:-cut, ind_ud_z], lw=1, color='k', linestyle=':', alpha=.9)
        ax1_t.fill_between(pii_cut, d2_noR[cut:-cut, ind_ld_z], d2_noR[cut:-cut, ind_ud_z],
                           lw=3, color='k', alpha=.1)
        ax1_t.plot(pii_cut, d2[cut:-cut, ind_med_z],linestyle='--', lw=3, color=colors[3])
        ax1_t.plot(pii_cut, d2[cut:-cut, ind_ud_z], lw=1, color=colors[3], linestyle=':')
        ax1_t.fill_between(pii_cut, d2[cut:-cut, ind_ld_z], d2[cut:-cut, ind_ud_z],
                           lw=3, color=colors[3], alpha=.2)

        ax1_t.axvline(pii[ind_med_r, 0], lw=1, linestyle='--', color='k', alpha=.6)
        ax1_t.set_xlim([0, 1])
        if left_top_ylim:
            ax1_t.set_ylim(left_top_ylim)
        ax1_t.set_xlabel(r"$R$", fontsize=15)



    def figure_1(self, cut=25, left_top_ylim=None, left_bottom_ylim=[-.0005, .0005], numb_lcurves=5, perc=.1):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_med_r = self.ind_med_r
        pii, zz = self.pii, self.zz
        d1_noR, d2_noR, d1, d2 = self.d1_noR, self.d2_noR, self.d1, self.d2
        mu_pii_noR, mu_pii, mu_pii_wc = self.mu_pii_noR, self.mu_pii, self.mu_pii_wc
        f_noR_dist, f_dist, f_wc_dist = self.f_noR_dist, self.f_dist, self.f_wc_dist
        alpha_z_tilde, kappa_tilde = self.alpha_z_tilde, self.kappa_tilde

        xmin, ymin = .0, -.8
        xmax, ymax = 1., .8

        f = plt.figure(figsize = (10, 5))
        gs0 = gsp.GridSpec(1, 7)

        gs00 = gsp.GridSpecFromSubplotSpec(9, 1, subplot_spec=gs0[:3])
        ax1_t = plt.Subplot(f, gs00[:, :], xlim=(xmin, xmax))
        #ax1_b = plt.Subplot(f, gs00[5:, :], xlim=(xmin, xmax))

        gs01 = gsp.GridSpecFromSubplotSpec(8, 8, subplot_spec=gs0[3:])
        ax2_t = plt.Subplot(f, gs01[:2, :-2], yticks=[], xticks=[], xlim=(xmin, xmax))
        ax2_r = plt.Subplot(f, gs01[2:, -2:], xticks=[], yticks=[], ylim=(ymin, ymax))
        ax2_c = plt.Subplot(f, gs01[2:, :-2], xlim=(xmin, xmax), ylim=(ymin, ymax))

        #====================================================================================
        #  LEFT PANELS
        #====================================================================================
        pii_cut = pii[cut:-cut, 0]

        ax1_t.plot(pii_cut, d1_noR[cut:-cut, ind_med_z], lw=3, color='k', label='without robustness')
        ax1_t.plot(pii_cut, d1_noR[cut:-cut, ind_ud_z], lw=1, color='k', linestyle=':', alpha=.5)
        ax1_t.fill_between(pii_cut, d1_noR[cut:-cut, ind_ld_z], d1_noR[cut:-cut, ind_ud_z],
                           lw=3, color='k', alpha=.1)
        ax1_t.plot(pii_cut, d1[cut:-cut, ind_med_z], lw=3, color=colors[3], linestyle='--', label='robust control')
        ax1_t.plot(pii_cut, d1[cut:-cut, ind_ud_z], lw=1, color=colors[3], linestyle='-.', alpha=.5)
        ax1_t.fill_between(pii_cut, d1[cut:-cut, ind_ld_z], d1[cut:-cut, ind_ud_z],
                           lw=3, color=colors[3], alpha=.2)
        ax1_t.axvline(pii[ind_med_r, 0], lw=1, linestyle='--', color='k', alpha=.6)
        ax1_t.set_xlim([0, 1])
        ax1_t.set_ylabel(r'$d^*_1$', fontsize=13, rotation=0)
        ax1_t.yaxis.set_label_coords(-0.18, 0.5)
        ax1_t.set_title("Investment ratio of the first capital", fontsize=13, y=1.02)
        ax1_t.legend(loc='best', fontsize=10)
        if left_top_ylim:
            ax1_t.set_ylim(left_top_ylim)

        # ax1_b.plot(pii[:, 0], mu_pii_noR[:, ind_med_z], color='k', lw=3, label="without robustness")
        # ax1_b.plot(pii[:, 0], mu_pii_noR[:, ind_ud_z], lw=1, color='k', linestyle=':', alpha=.5)
        # ax1_b.fill_between(pii[:, 0], mu_pii_noR[:, ind_ld_z], mu_pii_noR[:, ind_ud_z], color='k', alpha=.1, lw=3)
        # ax1_b.plot(pii[:, 0], mu_pii[:, ind_med_z], color=colors[3], lw=3,
        #            label="robust control under baseline")
        # ax1_b.plot(pii[:, 0], mu_pii[:, ind_ud_z], lw=1, color=colors[3], linestyle=':', alpha=.5)
        # ax1_b.fill_between(pii[:, 0], mu_pii[:, ind_ld_z], mu_pii[:, ind_ud_z], color=colors[3], alpha=.2, lw=3)
        # ax1_b.plot(pii[:, 0], mu_pii_wc[:, ind_med_z], color=colors[0], alpha=.8, lw=3,
        #            label="robust control under worst-case")
        # ax1_b.plot(pii[:, 0], mu_pii_wc[:, ind_ud_z], lw=1, color=colors[0], linestyle=':', alpha=.5)
        # ax1_b.fill_between(pii[:, 0], mu_pii_wc[:, ind_ld_z], mu_pii_wc[:, ind_ud_z],
        #                color=colors[0], alpha=.2, lw=3)
        # ax1_b.axhline(0, color='k', lw=1, linestyle='--')
        # ax1_b.axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
        # ax1_b.set_ylabel(r'$\mu_R$', fontsize=15, rotation=0)
        # ax1_b.yaxis.set_label_coords(-0.18, 0.5)
        # ax1_b.set_title("Local mean of the capital ratio", fontsize=15, y=1.01)
        # ax1_b.set_xlabel(r'$R$', fontsize=15)
        # ax1_b.set_ylim(left_bottom_ylim)
        #ax1_b.legend(loc=3, fontsize=12)


        #====================================================================================
        #  RIGHT PANELS
        #====================================================================================
        levels = np.linspace(confidence_band(f_noR_dist, perc), f_noR_dist.max(), numb_lcurves)
        ax2_c.contour(pii[1:, :], zz[1:, :], f_noR_dist, levels, colors='k', alpha=.6)
        ax2_c.contourf(pii[1:, :], zz[1:, :], f_noR_dist, [confidence_band(f_noR_dist, perc), 1],
                       alpha=.1, colors='k')

        levels = np.linspace(confidence_band(f_dist, perc), f_dist.max(), numb_lcurves)
        ax2_c.contour(pii[1:, :], zz[1:, :], f_dist, levels, colors=[colors[3]], alpha=.6, linestyles=['dashed'], linewidths=2)
        ax2_c.contourf(pii[1:, :], zz[1:, :], f_dist, [confidence_band(f_dist, perc), 1],
                       alpha=.2, colors=[colors[3]])

        levels = np.linspace(confidence_band(f_wc_dist, perc), f_wc_dist.max(), numb_lcurves)
        ax2_c.contour(pii[1:, :], zz[1:, :], f_wc_dist, levels, colors=[colors[0]], alpha=.6, linestyles=['dotted'], linewidths=2)
        ax2_c.contourf(pii[1:, :], zz[1:, :], f_wc_dist, [confidence_band(f_wc_dist, perc), 1],
                       alpha=.2, colors=[colors[0]])
        ax2_c.axhline(0, lw=1, linestyle='--', color='k', alpha=.6)
        #ax2_c.axhline(self.alpha_z_tilde/self.kappa_hat, lw=2, linestyle="-.", color=colors[2], alpha=.7)
        ax2_c.axhline(-.005/.017, lw=2, linestyle="-.", color=colors[2], alpha=.7)
        ax2_c.axvline(pii[ind_med_r, 0], lw=1, linestyle='--', color='k', alpha=.6)
        ax2_c.set_ylabel(r'$Z$', fontsize=15, rotation=0)
        ax2_c.set_xlabel(r'$R$', fontsize=15)
        ax2_c.plot(0, 0, label='without robustness', color='k', alpha=.7)
        ax2_c.plot(0, 0, label='robust control under baseline', color=colors[3], alpha=.7)
        ax2_c.plot(0, 0, label='robust control under worst-case', color=colors[0], alpha=.7)
        ax2_c.legend(fontsize=10)

        ax2_t.plot(pii[1:, 0], f_noR_dist.sum(1), color='k', lw=2, alpha=.7)
        ax2_t.fill_between(pii[1:, 0], 0, f_noR_dist.sum(1), color='k', alpha=.1)
        ax2_t.plot(pii[1:, 0], f_dist.sum(1), color=colors[3], lw=2, alpha=.7, linestyle='--')
        ax2_t.fill_between(pii[1:, 0], 0, f_dist.sum(1), color=colors[3], alpha=.2)
        ax2_t.plot(pii[1:, 0], f_wc_dist.sum(1), color=colors[0], lw=2, alpha=.7, linestyle=':')
        ax2_t.fill_between(pii[1:, 0], 0, f_wc_dist.sum(1), color=colors[0], alpha=.2)
        ax2_t.axvline(pii[ind_med_r, 0], lw=1, linestyle='--', color='k', alpha=.6)
        ax2_t.set_ylim([0, max(f_dist.sum(1).max(), f_wc_dist.sum(1).max())*1.2])
        ax2_t.set_xlim([0, 1])
        ax2_t.set_title("Stationary distributions of the states", fontsize=15, y=1.05)

        J_half = int(self.J/2)

        ax2_r.plot(f_noR_dist.sum(0), zz[0, :], color='k', lw=2, alpha=.4)
        ax2_r.fill_between(f_noR_dist.sum(0)[:J_half+1], zz[0, :J_half+1], zz[0, J_half:][::-1], facecolor='k', alpha=.1)
        ax2_r.plot(f_dist.sum(0), zz[0, :], color=colors[3], lw=2, alpha=.7, linestyle='--')
        ax2_r.fill_between((f_dist.sum(0))[:J_half+1], zz[0, :J_half+1], zz[0, J_half:][::-1], color=colors[3], alpha=.2)
        ax2_r.plot(f_wc_dist.sum(0), zz[0, :], color=colors[0], lw=2, alpha=.7, linestyle=':')
        ax2_r.fill_between((f_wc_dist.sum(0))[:-1], zz[0, -1], zz[0, :-1], color=colors[0], alpha=.2)
        ax2_r.axhline(0.0, lw=1, linestyle='--', color='k', alpha=.6)
        ax2_r.axhline(-.005/.017, lw=2, linestyle="-.", color=colors[2], alpha=.7)
        #ax2_r.axhline(alpha_z_tilde/self.kappa_hat, lw=2, linestyle="-.", color=colors[2], alpha=.7)
        ax2_r.set_ylim([ymin, ymax])
        ax2_r.set_xlim([0, max(f_dist.sum(0).max(), f_wc_dist.sum(0).max())*1.2])

        f.add_subplot(ax1_t)
        #f.add_subplot(ax1_b) 
        f.add_subplot(ax2_t)
        f.add_subplot(ax2_r)
        f.add_subplot(ax2_c)

        f.tight_layout()
        return f


    #def irf_figure(self, cut=25, left_top_ylim=None, left_bottom_ylim=[-.0005, .0005],
    #             numb_lcurves=5, perc=.1):


    def figure_1_components(self, ax2_t, ax2_c, ax2_r, left=False, numb_lcurves=5, perc=.1):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_med_r = self.ind_med_r
        pii, zz = self.pii, self.zz
        f_noR_dist, f_dist, f_wc_dist = self.f_noR_dist, self.f_dist, self.f_wc_dist
        alpha_z_tilde, kappa_tilde = self.alpha_z_tilde, self.kappa_tilde

        #====================================================================================
        #  RIGHT PANELS
        #====================================================================================
        levels = np.linspace(confidence_band(f_noR_dist, perc), f_noR_dist.max(), numb_lcurves)
        ax2_c.contour(pii[1:, :], zz[1:, :], f_noR_dist, levels, colors='k', alpha=.6)
        ax2_c.contourf(pii[1:, :], zz[1:, :], f_noR_dist, [confidence_band(f_noR_dist, perc), 1],
                       alpha=.1, colors='k')

        levels = np.linspace(confidence_band(f_dist, perc), f_dist.max(), numb_lcurves)
        ax2_c.contour(pii[1:, :], zz[1:, :], f_dist, levels, colors=[colors[3]], alpha=.6)
        ax2_c.contourf(pii[1:, :], zz[1:, :], f_dist, [confidence_band(f_dist, perc), 1],
                       alpha=.2, colors=[colors[3]])

        levels = np.linspace(confidence_band(f_wc_dist, perc), f_wc_dist.max(), numb_lcurves)
        ax2_c.contour(pii[1:, :], zz[1:, :], f_wc_dist, levels, colors=[colors[0]], alpha=.6)
        ax2_c.contourf(pii[1:, :], zz[1:, :], f_wc_dist, [confidence_band(f_wc_dist, perc), 1],
                       alpha=.2, colors=[colors[0]])
        ax2_c.axhline(0, lw=1, linestyle='--', color='k', alpha=.6)
        ax2_c.axhline(self.alpha_z_tilde/self.kappa_hat, lw=2, linestyle="-.", color=colors[2], alpha=.7)
        ax2_c.axhline(-.005/.017, lw=2, linestyle=":", color=colors[1], alpha=.7)
        ax2_c.axvline(pii[ind_med_r, 0], lw=1, linestyle='--', color='k', alpha=.6)
        ax2_c.set_xlabel(r'$R$', fontsize=15)
        ax2_c.plot(0, 0, label='without robustness', color='k', alpha=.7)
        ax2_c.plot(0, 0, label='robust control under baseline', color=colors[3], alpha=.7)
        ax2_c.plot(0, 0, label='robust control under worst-case', color=colors[0], alpha=.7)
        ax2_c.legend(fontsize=12)

        ax2_t.plot(pii[1:, 0], f_noR_dist.sum(1), color='k', lw=5, alpha=.7)
        ax2_t.fill_between(pii[1:, 0], 0, f_noR_dist.sum(1), color='k', alpha=.1)
        ax2_t.plot(pii[1:, 0], f_dist.sum(1), color=colors[3], lw=2, alpha=.7)
        ax2_t.fill_between(pii[1:, 0], 0, f_dist.sum(1), color=colors[3], alpha=.2)
        ax2_t.plot(pii[1:, 0], f_wc_dist.sum(1), color=colors[0], lw=2, alpha=.7)
        ax2_t.fill_between(pii[1:, 0], 0, f_wc_dist.sum(1), color=colors[0], alpha=.2)
        ax2_t.axvline(pii[ind_med_r, 0], lw=1, linestyle='--', color='k', alpha=.6)
        ax2_t.set_ylim([0, max(f_dist.sum(1).max(), f_wc_dist.sum(1).max())*1.2])
        ax2_t.set_xlim([0, 1])

        if left:
            ax2_r.plot(-f_noR_dist.sum(0), zz[0, :], color='k', lw=2, alpha=.4)
            ax2_r.fill_between(-f_noR_dist.sum(0)[:76], zz[0, :76], zz[0, 75:][::-1], facecolor='k', alpha=.1)
            ax2_r.plot(-f_dist.sum(0), zz[0, :], color=colors[3], lw=2, alpha=.7)
            ax2_r.fill_between((-f_dist.sum(0))[:76], zz[0, :76], zz[0, 75:][::-1], color=colors[3], alpha=.2)
            ax2_r.plot(-f_wc_dist.sum(0), zz[0, :], color=colors[0], lw=2, alpha=.7)
            ax2_r.fill_between((-f_wc_dist.sum(0))[:-1], zz[0, -1], zz[0, :-1], color=colors[0], alpha=.2)
            ax2_r.axhline(0.0, lw=1, linestyle='--', color='k', alpha=.6)
            ax2_r.axhline(-.005/.017, lw=2, linestyle=":", color=colors[1], alpha=.7)
            ax2_r.axhline(alpha_z_tilde/self.kappa_hat, lw=2, linestyle="-.", color=colors[2], alpha=.7)
            ax2_r.set_ylim([ymin, ymax])
            ax2_r.set_xlim([-max(f_dist.sum(0).max(), f_wc_dist.sum(0).max())*1.2, 0])

        else:
            ax2_r.plot(f_noR_dist.sum(0), zz[0, :], color='k', lw=2, alpha=.4)
            ax2_r.fill_between(f_noR_dist.sum(0)[:76], zz[0, :76], zz[0, 75:][::-1], facecolor='k', alpha=.1)
            ax2_r.plot(f_dist.sum(0), zz[0, :], color=colors[3], lw=2, alpha=.7)
            ax2_r.fill_between((f_dist.sum(0))[:76], zz[0, :76], zz[0, 75:][::-1], color=colors[3], alpha=.2)
            ax2_r.plot(f_wc_dist.sum(0), zz[0, :], color=colors[0], lw=2, alpha=.7)
            ax2_r.fill_between((f_wc_dist.sum(0))[:-1], zz[0, -1], zz[0, :-1], color=colors[0], alpha=.2)
            ax2_r.axhline(0.0, lw=1, linestyle='--', color='k', alpha=.6)
            ax2_r.axhline(-.005/.017, lw=2, linestyle=":", color=colors[1], alpha=.7)
            ax2_r.axhline(alpha_z_tilde/self.kappa_hat, lw=2, linestyle="-.", color=colors[2], alpha=.7)
            ax2_r.set_ylim([ymin, ymax])
            ax2_r.set_xlim([0, max(f_dist.sum(0).max(), f_wc_dist.sum(0).max())*1.2])


        f.tight_layout()



    def figure_drift(self, cdrift_ylim=[-.1, .9], zdrift_ylim=[-.017, .01]):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_ld_r, ind_med_r, ind_ud_r = self.ind_ld_r, self.ind_med_r, self.ind_ud_r
        pii, zz = self.pii, self.zz
        inner = self.inner
        logC_mu_noR, logC_mu, logC_mu_wc = 100*self.logC_mu_noR, 100*self.logC_mu, 100*self.logC_mu_wc
        mu_z, mu_z_wc = self.mu_z, self.mu_z_wc

        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        zz_cut = zz[0, ind_ld_z:ind_ud_z]

        ax[0].plot(zz_cut, logC_mu_noR[ind_med_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                   color='k', lw=3, label="without robustness")
        ax[0].fill_between(zz_cut, logC_mu_noR[ind_ld_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                                   logC_mu_noR[ind_ud_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                           color='k', alpha=.1, lw=3)

        ax[0].plot(zz_cut, logC_mu[ind_med_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                   color=colors[3], lw=3, label="robust control under baseline")
        ax[0].fill_between(zz_cut, logC_mu[ind_ld_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                                   logC_mu[ind_ud_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                           color=colors[3], alpha=.2, lw=3)

        ax[0].plot(zz_cut, logC_mu_wc[ind_med_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                   color=colors[0], alpha=.8, lw=3, label="robust control under worst-case")
        ax[0].fill_between(zz_cut, logC_mu_wc[ind_ld_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                                   logC_mu_wc[ind_ud_r-inner, ind_ld_z-inner:ind_ud_z-inner],
                           color=colors[0], alpha=.2, lw=3)
        ax[0].legend(loc=2)
        ax[0].axhline(self.alpha_c_hat, lw=1, linestyle='--', color='k', alpha=.6)
        ax[0].axvline(0, lw=1, linestyle='--', color='k', alpha=.6)
        ax[0].set_ylim(cdrift_ylim)
        ax[0].set_xlim([-0.3, .3])
        ax[0].set_xlabel(r"$Z$")
        ax[0].set_title(r"$E_t\left[{\log C}\right]$")


        ax[1].plot(zz_cut, mu_z[ind_med_r, ind_ld_z:ind_ud_z], color='k', lw=3, label="under baseline")
        ax[1].fill_between(zz_cut, mu_z[ind_ld_r, ind_ld_z:ind_ud_z], mu_z[ind_ud_r, ind_ld_z:ind_ud_z],
                           color='k', alpha=.1, lw=3)
        ax[1].plot(zz_cut, mu_z_wc[ind_med_r, ind_ld_z:ind_ud_z], color=colors[0],
                   alpha=.8, lw=3, label="under worst-case")
        ax[1].fill_between(zz_cut, mu_z_wc[ind_ld_r, ind_ld_z:ind_ud_z], mu_z_wc[ind_ud_r, ind_ld_z:ind_ud_z],
                           color=colors[0], alpha=.2, lw=3)
        ax[1].axhline(0, lw=1, linestyle='--', color='k', alpha=.6)
        ax[1].axvline(0, lw=1, linestyle='--', color='k', alpha=.6)
        ax[1].legend(loc='best')
        ax[1].set_ylim(zdrift_ylim)
        ax[1].set_xlim([-0.3, .3])
        ax[1].set_xlabel(r"$Z$")
        ax[1].set_title(r"$\mu_Z(R, Z)$")

        plt.tight_layout()



    def figure_H(self, ax1, ax2, ylim=None, sharey=False):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_ld_r, ind_med_r, ind_ud_r = self.ind_ld_r, self.ind_med_r, self.ind_ud_r
        h1_dist, h2_dist, hz_dist = self.h1_dist, self.h2_dist, self.hz_dist
        pii, rr, zz = self.pii, self.rr, self.zz

        inn_r = slice(ind_ld_r, ind_ud_r + 2)
        inn_z = slice(ind_ld_z, ind_ud_z + 1)

        if not np.isclose(h1_dist[0, :], 0).all():
            ax1.plot(pii[inn_r, 1], h1_dist[inn_r, ind_med_z], label=r'$h^*_1$', lw=2, color=colors[1])
            ax1.plot(pii[inn_r, 1], h1_dist[inn_r, ind_ud_z], linestyle=':', lw=2, color=colors[1])
            ax1.plot(pii[inn_r, 1], h1_dist[inn_r, ind_ld_z], linestyle='--', lw=2, color=colors[1])
            ax1.fill_between(pii[inn_r, 1], h1_dist[inn_r, ind_ld_z], h1_dist[inn_r, ind_ud_z],
                               alpha=.1, color=colors[1])

        ax1.plot(pii[inn_r, 1], h2_dist[inn_r, ind_med_z], label=r'$h^*_2$', lw=2, color=colors[0])
        ax1.plot(pii[inn_r, 1], h2_dist[inn_r, ind_ud_z], linestyle=':', lw=2, color=colors[0])
        ax1.plot(pii[inn_r, 1], h2_dist[inn_r, ind_ld_z], linestyle='--', lw=2, color=colors[0])
        ax1.fill_between(pii[inn_r, 1], h2_dist[inn_r, ind_ld_z], h2_dist[inn_r, ind_ud_z],
                           alpha=.1, color=colors[0])

        if not np.isclose(h1_dist[0, :], 0).all():
            ax1.legend(loc='best', fontsize=13)

        ax1.axvline(pii[ind_med_r, 1], lw=1, linestyle='--', color='k')

        if ylim:
            ax2.set_ylim(ylim)
        ax2.plot(pii[inn_r, 1], hz_dist[inn_r, ind_med_z], label=r'median $R$', lw=2, color=colors[3])
        ax2.plot(pii[inn_r, 1], hz_dist[inn_r, ind_ud_z], label=r'$.9$ decile', lw=2,
                   linestyle=':', color=colors[3])
        ax2.plot(pii[inn_r, 1], hz_dist[inn_r, ind_ld_z], label=r'$.1$ decile', lw=2,
                   linestyle='--', color=colors[3])
        ax2.fill_between(pii[inn_r, 1], hz_dist[inn_r, ind_ld_z], hz_dist[inn_r, ind_ud_z],
                           alpha=.1, color=colors[3])
        ax2.axvline(pii[ind_med_r, 1], lw=1, linestyle='--', color='k')
        ax2.set_xlabel(r"$R$", fontsize=15)



    def figure_U(self, ax1, ax2, ylim=None, ylim_top=None, sharey=False):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_ld_r, ind_med_r, ind_ud_r = self.ind_ld_r, self.ind_med_r, self.ind_ud_r
        h1_dist, h2_dist, hz_dist = self.h1_dist, self.h2_dist, self.hz_dist
        pii, rr, zz = self.pii, self.rr, self.zz

        #inn_r = slice(ind_ld_r, ind_ud_r + 2)
        inn_r = slice(5, -5)
        inn_z = slice(ind_ld_z, ind_ud_z + 1)

        if not np.isclose(h1_dist[0, :], 0).all():
            u12 = -(h1_dist + h2_dist)/np.sqrt(2)
            ax1.plot(pii[inn_r, 1], u12[inn_r, ind_med_z], lw=2, color=colors[0])
            ax1.plot(pii[inn_r, 1], u12[inn_r, ind_ud_z], linestyle=':', lw=2, color=colors[0])
            ax1.plot(pii[inn_r, 1], u12[inn_r, ind_ld_z], linestyle='--', lw=2, color=colors[0])
            ax1.fill_between(pii[inn_r, 1], u12[inn_r, ind_ld_z], u12[inn_r, ind_ud_z],
                               alpha=.1, color=colors[0])
        else:
            u2 = -h2_dist
            ax1.plot(pii[inn_r, 1], u2[inn_r, ind_med_z], label=r'$h^*_2$', lw=2, color=colors[0])
            ax1.plot(pii[inn_r, 1], u2[inn_r, ind_ud_z], linestyle=':', lw=2, color=colors[0])
            ax1.plot(pii[inn_r, 1], u2[inn_r, ind_ld_z], linestyle='--', lw=2, color=colors[0])
            ax1.fill_between(pii[inn_r, 1], u2[inn_r, ind_ld_z], u2[inn_r, ind_ud_z],
                           alpha=.1, color=colors[0])

        if ylim_top:
            ax1.set_ylim(ylim_top)

        ax1.axvline(pii[ind_med_r, 1], lw=1, linestyle='--', color='k')

        uz = -hz_dist
        if ylim:
            ax2.set_ylim(ylim)
        ax2.plot(pii[inn_r, 1], uz[inn_r, ind_med_z], label=r'median $R$', lw=2, color=colors[3])
        ax2.plot(pii[inn_r, 1], uz[inn_r, ind_ud_z], label=r'$.9$ decile', lw=2,
                   linestyle=':', color=colors[3])
        ax2.plot(pii[inn_r, 1], uz[inn_r, ind_ld_z], label=r'$.1$ decile', lw=2,
                   linestyle='--', color=colors[3])
        ax2.fill_between(pii[inn_r, 1], uz[inn_r, ind_ld_z], uz[inn_r, ind_ud_z],
                           alpha=.1, color=colors[3])
        ax2.axvline(pii[ind_med_r, 1], lw=1, linestyle='--', color='k')
        ax2.set_xlabel(r"$R$", fontsize=15)


    def figure_states(self, ylim=[-.005, .005], xlim=None):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_ld_r, ind_med_r, ind_ud_r = self.ind_ld_r, self.ind_med_r, self.ind_ud_r
        pii, rr, zz = self.pii, self.rr, self.zz
        mu_pii_noR, mu_pii, mu_pii_wc = self.mu_pii_noR, self.mu_pii, self.mu_pii_wc
        mu_z, mu_z_wc = self.mu_z, self.mu_z_wc

        inn_r = slice(ind_ld_r, ind_ud_r + 2)
        inn_z = slice(ind_ld_z, ind_ud_z + 1)


        fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        ax[0, 0].plot(pii[inn_r, 0], mu_pii_noR[inn_r, ind_med_z], color='k', lw=3, label="without robustness")
        ax[0, 0].fill_between(pii[inn_r, 0], mu_pii_noR[inn_r, ind_ld_z], mu_pii_noR[inn_r, ind_ud_z],
                              color='k', alpha=.1, lw=3)
        ax[0, 0].plot(pii[inn_r, 0], mu_pii[inn_r, ind_med_z], color=colors[3], lw=3,
                      label="robust control under baseline")
        ax[0, 0].fill_between(pii[inn_r, 0], mu_pii[inn_r, ind_ld_z], mu_pii[inn_r, ind_ud_z],
                              color=colors[3], alpha=.2, lw=3)
        ax[0, 0].plot(pii[inn_r, 0], mu_pii_wc[inn_r, ind_med_z], color=colors[0], alpha=.8, lw=3,
                 label="robust control under worst-case")
        ax[0, 0].fill_between(pii[inn_r, 0], mu_pii_wc[inn_r, ind_ld_z], mu_pii_wc[inn_r, ind_ud_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[0, 0].axhline(0, color='k', lw=1, linestyle='--')
        ax[0, 0].axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
        ax[0, 0].set_ylabel(r'$\mu_R$', fontsize=14, rotation=0)
        ax[0, 0].yaxis.set_label_coords(-0.14, 0.5)
        ax[0, 0].set_ylim(ylim)
        if xlim:
            ax[0, 0].set_xlim(xlim)
        ax[0, 0].legend(loc='best')

        ax[0, 1].plot(zz[0, inn_z], mu_pii_noR[ind_med_r, inn_z], color='k', lw=3, label="without robustness")
        ax[0, 1].fill_between(zz[0, inn_z], mu_pii_noR[ind_ld_r, inn_z], mu_pii_noR[ind_ud_r, inn_z],
                              color='k', alpha=.1, lw=3)
        ax[0, 1].plot(zz[0, inn_z], mu_pii[ind_med_r, inn_z], color=colors[3], lw=3,
                   label="robust control under baseline")
        ax[0, 1].fill_between(zz[0, inn_z], mu_pii[ind_ld_r, inn_z], mu_pii[ind_ud_r, inn_z],
                              color=colors[3], alpha=.2, lw=3)
        ax[0, 1].plot(zz[0, inn_z], mu_pii_wc[ind_med_r, inn_z], color=colors[0], lw=3,
                   label="robust control under worst-case")
        ax[0, 1].fill_between(zz[0, inn_z], mu_pii_wc[ind_ld_r, inn_z], mu_pii_wc[ind_ud_r, inn_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[0, 1].axhline(0, color='k', lw=1, linestyle='--')
        ax[0, 1].axvline(0, color='k', lw=1, linestyle='--')
        ax[0, 1].set_ylim(ylim)
        ax[0, 1].set_xlim([-.25, .25])
        ax[0, 1].legend(loc='best')

        ax[1, 0].plot(pii[inn_r, 0], mu_z[inn_r, ind_med_z], color='k', lw=3, label="under baseline")
        ax[1, 0].fill_between(pii[inn_r, 0], mu_z[inn_r, ind_ld_z], mu_z[inn_r, ind_ud_z],
                              color='k', alpha=.1, lw=3)
        ax[1, 0].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_med_z], color=colors[0], alpha=.8, lw=3,
                 label="under worst-case")
        ax[1, 0].fill_between(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], mu_z_wc[inn_r, ind_ud_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[1, 0].axhline(0, color='k', lw=1, linestyle='--')
        ax[1, 0].axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
        ax[1, 0].set_ylabel(r'$\mu_Z$', fontsize=14, rotation=0)
        ax[1, 0].yaxis.set_label_coords(-0.14, 0.5)
        ax[1, 0].set_xlabel(r'$R$', fontsize=14)
        ax[1, 0].set_ylim([-.017, .01])
        if xlim:
            ax[0, 0].set_xlim(xlim)
        ax[1, 0].legend(loc=2)

        ax[1, 1].plot(zz[0, inn_z], mu_z[ind_med_r, inn_z], color='k', lw=3, label="under baseline")
        ax[1, 1].fill_between(zz[0, inn_z], mu_z[ind_ld_r, inn_z], mu_z[ind_ud_r, inn_z],
                              color='k', alpha=.1, lw=3)
        ax[1, 1].plot(zz[0, inn_z], mu_z_wc[ind_med_r, inn_z], color=colors[0], lw=3,
                   label="under worst-case")
        ax[1, 1].fill_between(zz[0, inn_z], mu_z_wc[ind_ld_r, inn_z], mu_z_wc[ind_ud_r, inn_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[1, 1].axhline(0, color='k', lw=1, linestyle='--')
        ax[1, 1].axvline(0, color='k', lw=1, linestyle='--')
        ax[1, 1].set_xlabel(r'$Z$', fontsize=14)
        ax[1, 1].set_ylim([-.017, .01])
        ax[1, 1].set_xlim([-.25, .25])
        ax[1, 1].legend(loc=2)

        plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)

        plt.tight_layout()

    def figure2_states(self, ylim_left=[-.005, .005], ylim_right=[-.17, .1], xlim=None):

        ind_ld_z, ind_med_z, ind_ud_z = self.ind_ld_z, self.ind_med_z, self.ind_ud_z
        ind_ld_r, ind_med_r, ind_ud_r = self.ind_ld_r, self.ind_med_r, self.ind_ud_r
        pii, rr, zz = self.pii, self.rr, self.zz
        mu_pii_noR, mu_pii, mu_pii_wc = self.mu_pii_noR, self.mu_pii, self.mu_pii_wc
        mu_z, mu_z_wc = self.mu_z, self.mu_z_wc

        inn_r = slice(ind_ld_r, ind_ud_r + 2)
        inn_z = slice(ind_ld_z, ind_ud_z + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

        ax[0].plot(zz[0, inn_z], mu_pii_noR[ind_med_r, inn_z], color='k', lw=3, label="without robustness")
        ax[0].plot(zz[0, inn_z], mu_pii_noR[ind_ld_r, inn_z], color='k', linestyle='--', lw=2, alpha=.4)
        ax[0].plot(zz[0, inn_z], mu_pii_noR[ind_ud_r, inn_z], color='k', linestyle=':',  lw=2, alpha=.4)
        ax[0].fill_between(zz[0, inn_z], mu_pii_noR[ind_ld_r, inn_z], mu_pii_noR[ind_ud_r, inn_z],
                              color='k', alpha=.1, lw=3)
        ax[0].plot(zz[0, inn_z], mu_pii[ind_med_r, inn_z], color=colors[3], lw=3,
                   label="robust control under baseline")
        ax[0].plot(zz[0, inn_z], mu_pii[ind_ld_r, inn_z], color=colors[3], linestyle='--', lw=2, alpha=.5)
        ax[0].plot(zz[0, inn_z], mu_pii[ind_ud_r, inn_z], color=colors[3], linestyle=':',  lw=2, alpha=.5)
        ax[0].fill_between(zz[0, inn_z], mu_pii[ind_ld_r, inn_z], mu_pii[ind_med_r, inn_z],
                              color=colors[3], alpha=.2, lw=3)
        ax[0].fill_between(zz[0, inn_z], mu_pii[ind_med_r, inn_z], mu_pii[ind_ud_r, inn_z],
                              color=colors[3], alpha=.2, lw=3)
        ax[0].plot(zz[0, inn_z], mu_pii_wc[ind_med_r, inn_z], color=colors[0], lw=3,
                   label="robust control under worst-case")
        ax[0].plot(zz[0, inn_z], mu_pii_wc[ind_ld_r, inn_z], color=colors[0], linestyle='--', lw=2, alpha=.5)
        ax[0].plot(zz[0, inn_z], mu_pii_wc[ind_ud_r, inn_z], color=colors[0], linestyle=':',  lw=2, alpha=.5)
        ax[0].fill_between(zz[0, inn_z], mu_pii_wc[ind_ld_r, inn_z], mu_pii_wc[ind_med_r, inn_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[0].fill_between(zz[0, inn_z], mu_pii_wc[ind_med_r, inn_z], mu_pii_wc[ind_ud_r, inn_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[0].axhline(0, color='k', lw=1, linestyle='--')
        ax[0].axvline(0, color='k', lw=1, linestyle='--')
        ax[0].set_title(r'$\mu_R(Z, R)$', fontsize=15)
        ax[0].set_xlabel(r'$Z$', fontsize=15)
        ax[0].set_ylim(ylim_left)
        if xlim:
            ax[0].set_xlim(xlim)
        ax[0].legend(loc=2, fontsize=12)

        ax[1].plot(pii[inn_r, 0], mu_z[inn_r, ind_med_z], color='k', lw=3, label="under baseline")
        ax[1].plot(pii[inn_r, 0], mu_z[inn_r, ind_ld_z], color='k', lw=2, linestyle='--', alpha=.4)
        ax[1].plot(pii[inn_r, 0], mu_z[inn_r, ind_ud_z], color='k', lw=2, linestyle=':', alpha=.4)
        ax[1].fill_between(pii[inn_r, 0], mu_z[inn_r, ind_ld_z], mu_z[inn_r, ind_ud_z],
                              color='k', alpha=.1, lw=3)
        ax[1].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_med_z], color=colors[0], alpha=.8, lw=3,
                 label="under worst-case")
        ax[1].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], color=colors[0], linestyle='--', lw=2, alpha=.5)
        ax[1].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_ud_z], color=colors[0], linestyle=':', lw=2, alpha=.5)
        ax[1].fill_between(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], mu_z_wc[inn_r, ind_ud_z],
                              color=colors[0], alpha=.2, lw=3)
        ax[1].axhline(0, color='k', lw=1, linestyle='--')
        ax[1].axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
        ax[1].set_title(r'$\mu_Z(R, Z)$', fontsize=15)
        ax[1].set_xlabel(r'$R$', fontsize=15)
        ax[1].set_ylim(ylim_right)
        ax[1].legend(loc=2, fontsize=12)


        plt.tight_layout()


    def tables(self, model_singlecapital):

        m = model_singlecapital
        title_row1 = " kappa_tilde |     q    |  az_tilde ||  alpha_c  |   beta   "
        title_row2 = "|   alpha_z   |   kappa    |     dc   "
        lines = "-------------------------------------------------------"
        col = "&   {: 0.3f}  "

        print('TABLE 1')
        print(title_row1 + title_row2)
        print(lines + lines)
        for i in range(m.shape[0]):
            rows = col * 7 + "&  {: 0.3f} \\\\"
            if i==2:
                print('\n')
            print(rows.format(m[i, 2], m[i, 0], m[i, 1], m[i, 3], m[i, 4], m[i, 5], m[i, 6], m[i, 7]))

        print('\n')
        print('TABLE 2')
        title_row = "||    q   |    RE   |  Chernoff |   HL  "
        cells = "&  {: 0.3f} &  {: 0.3f}  &  {: 0.3f} &  {: 3.0f} "
        print("  kappa  |  alpha_z  " + title_row + title_row)
        print("-----------------------------------------------------------------------------")
        for i in range(2, m.shape[0]):
            rows ="& {: 0.3f}  &  {: 0.3f}  " + cells + cells + "\\\\"
            print(rows.format(m[i, 2], m[i, 1], m[i, 0], m[i, 8], m[i, 10], m[i, 11],
                              m[i, 12], m[i, 13], m[i, 14], m[i, 15]))

    def tables2(self, model_singlecapital):

        sigma_z = self.sigma_z_1cap
        s2 = np.dot(sigma_z, sigma_z)

        m = model_singlecapital
        title_row1 = "    q   |  az_til |  k_til | b_til || alpha_c | beta "
        title_row2 = "|  alpha_z |  kappa  |  dc   |  Ez  |  sigma(z) "
        lines = "-------------------------------------------------------"
        col = "& {: 0.3f} "

        print('TABLE 1')
        print(title_row1 + title_row2)
        print(lines + lines)
        for i in range(m.shape[0]):
            rows = col * 10 + "& {: 0.3f} \\\\"
            if i==2 or i==8:
                print('\n')
            print(rows.format(m[i, 0], m[i, 1], m[i, 2], m[i, 3], m[i, 4], m[i, 5], m[i, 6], m[i, 7], m[i, 8], m[i, 6]/m[i, 7], np.sqrt(s2/(2*m[i, 7])) ))

        print('\n')
        print('TABLE 2')
        title_row = "||    q   |    RE   |  Chernoff |   HL  "
        cells = "&  {: 0.3f} &  {: 0.3f}  &  {: 0.3f} &  {: 3.0f} "
        print("  kappa  |  beta   |  alpha_z  " + title_row + title_row)
        print("-----------------------------------------------------------------------------")
        for i in range(2, m.shape[0]):
            rows ="& {: 0.3f}  &  {: 0.3f} &  {: 0.3f} " + cells + cells + "\\\\"
            print(rows.format(m[i, 2], m[i, 3], m[i, 1], m[i, 0], m[i, 9], m[i, 11], m[i, 12],
                              m[i, 13], m[i, 14], m[i, 15], m[i, 16]))


class plottingmodule():
    def __init__(self):
        tilt_graph = pickle.load(open('./data/plotdata_12.pickle', "rb", -1))
        self.sets_const_tilt = tilt_graph['sets_const']
        self.sets_quad_tilt = tilt_graph['sets_quad']
        self.worstcase_intercept = tilt_graph['worstcase_intercept']
        self.worstcase_intercept_path = tilt_graph['wc_path_intercept']
        self.worstcase_persistence_path = tilt_graph['wc_path_persistence']
        self.worstcase_persistence = tilt_graph['worstcase_persistence']
        self.isos_intercept = tilt_graph['isos_intercept']
        self.isos_persistence = tilt_graph['isos_persistence']
        self.isos_intercept_val = tilt_graph['isos_intercept_val']
        self.isos_persistence_val = tilt_graph['isos_persistence_val']

        self.ex_post = pickle.load(open('./data/plotdata_4.pickle', "rb", -1))
        self.kappa_hat_list = sorted(self.ex_post.keys())
        self.ex_ante = pickle.load(open('./data/plotdata_6.pickle', "rb", -1))
        self.beta_hat_list = sorted(self.ex_ante.keys())

        self.twistingfunction = pickle.load(open('./data/plotdata_3.pickle', "rb", -1))
        self.twisting_list = sorted(list(self.twistingfunction.keys()))

        temp_dict = pickle.load(open('./data/plotdata_57.pickle', "rb", -1))
        self.sym = temp_dict['sym']
        self.asym = temp_dict['asym']
        self.kappa_hat_list1 = sorted(self.sym.keys())
        self.beta_hat_list1 = sorted(self.asym.keys())

        self.shock_densities = pickle.load(open('./data/plotdata_9.pickle', "rb", -1))

    def const_tilt_plot(self):
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        models = sorted(list(self.sets_const_tilt.keys()),reverse = True)
        fig = go.Figure()
        ite = 0
        for kappa, alpha in models:
            if kappa == 0.169 and alpha == 0:
                fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = True, marker=dict(size = 8), name = r"$\text{{baseline: }}(\hat{{\alpha}}, \hat{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), showlegend = True, legendgroup = 'Baseline Model', mode = 'markers'))
            else:
                if ite == 0: 
                    fig.add_trace(go.Scatter(x = [kappa], y = [alpha], marker=dict(color='LightSkyBlue', size = 8), visible = False, name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), showlegend = True, legendgroup = 'Worrisome Model', mode = 'markers'))

                    fig.add_trace(go.Scatter(x = self.sets_const_tilt[kappa,alpha][:,0], y = self.sets_const_tilt[kappa,alpha][:,1], visible = False, fill = 'toself', mode = 'lines', fillcolor = red_fill,
                        line = dict(color = red_line), showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))

                    ite = 1
                else:
                    fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = False, marker=dict(color='LightSkyBlue', size = 8), name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), showlegend = True, legendgroup = 'Worrisome Model', mode = 'markers'))

                    # fig.add_trace(go.Scatter(x = self.sets_const_tilt[kappa,alpha][:,0], y = y_down, visible = False, line = dict(color = red_line),
                    #     showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))
                    fig.add_trace(go.Scatter(x = self.sets_const_tilt[kappa,alpha][:,0], y = self.sets_const_tilt[kappa,alpha][:,1], visible = False, fill = 'toself', mode = 'lines', fillcolor = red_fill,
                        line = dict(color = red_line), showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))

        
        xs = np.linspace(0.02, 0.3, 50)
        ys = np.linspace(-0.065, 0.04, 50)
        fig.add_trace(go.Scatter(x = [0.169] * 50, y = ys, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))
        fig.add_trace(go.Scatter(x = xs, y = [0] * 50, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))

        fig.data[10 * 2]['visible'] = True
        fig.data[10 * 2 -1]['visible'] = True
        steps = []
        for i in range(len(models)):
            if i == 0:
                pass
            else:
                label = '{}'.format(models[i][1])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][i*2] = True
                step['args'][1][i*2 - 1] = True
                step['args'][1][-2] = True
                step['args'][1][-1] = True

                steps.append(step)


        sliders = [dict(active = 10,
                    currentvalue = {"prefix": 'alpha: '},
                    pad = {"t": len(models)},
                    steps = steps,
                    x = 0, y = 0.25)]

        fig.update_layout(title = dict(text = "Constant tilting function", font = dict(size = 20)),
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text=r'$\kappa$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text=r'$\alpha$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            sliders = sliders,
                            plot_bgcolor = 'rgba(0,0,0,0)',
                            legend = dict(x = 0, y = -0.3, orientation = 'h'),
                            width = 500,
                            height = 500,
                            margin = dict(l=5, r=20, t=40, b=10),
                            autosize = False
                            )
        fig.update_xaxes(range = [0.02, 0.3], showline = True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(range = [-.065, .04], showline = True, linewidth=2, linecolor='black', mirror = True)

        figw = go.FigureWidget(fig)
        return figw

    def quad_tilt_plot(self):
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        models = sorted(list(self.sets_quad_tilt.keys()),reverse = True)
        fig = go.Figure()
        ite = 0
        for kappa, alpha in models:
            if kappa == 0.169 and alpha == 0:
                fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = True, marker = dict(size = 8), name = r"$\text{{baseline: }}(\hat{{\alpha}}, \hat{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), showlegend = True, legendgroup = 'Baseline Model', mode = 'markers'))
            else:
                if ite == 0: 
                    fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = False, name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), showlegend = True, 
                            marker=dict(color='LightSkyBlue', size = 8), legendgroup = 'worrisome Model', mode = 'markers'))

                    fig.add_trace(go.Scatter(x = self.sets_quad_tilt[kappa,alpha][:,0], y = self.sets_quad_tilt[kappa,alpha][:,1], visible = False, line = dict(color = red_line), 
                        fill = 'toself', mode = 'lines', fillcolor = red_fill, showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))
                    ite = 1
                else:
                    fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = False, name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), showlegend = True, 
                            marker=dict(color='LightSkyBlue', size = 8), legendgroup = 'worrisome Model', mode = 'markers'))

                    fig.add_trace(go.Scatter(x = self.sets_quad_tilt[kappa,alpha][:,0], y = self.sets_quad_tilt[kappa,alpha][:,1], visible = False, line = dict(color = red_line),
                       fill = 'toself', mode = 'lines', fillcolor = red_fill, showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))

        xs = np.linspace(0.02, 0.3, 50)
        ys = np.linspace(-0.065, 0.04, 50)
        fig.add_trace(go.Scatter(x = [0.169] * 50, y = ys, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))
        fig.add_trace(go.Scatter(x = xs, y = [0] * 50, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))
        steps = []
        fig.data[10 * 2]['visible'] = True
        fig.data[10 * 2 -1]['visible'] = True

        for i in range(len(models)):
            if i == 0:
                pass
            else:
                label = '{}'.format(models[i][1])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][i*2] = True
                step['args'][1][i*2 - 1] = True
                step['args'][1][-1] = True
                step['args'][1][-2] = True

                steps.append(step)

        sliders = [dict(active = 10,
                    currentvalue = {"prefix": 'alpha: '},
                    pad = {"t": len(models)},
                    steps = steps,
                    x = 0,
                    y = 0.25)]

        fig.update_layout(title = dict(text = "Quadratic tilting function", font = dict(size = 20)),
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text=r'$\kappa$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text=r'$\alpha$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            sliders = sliders,
                            plot_bgcolor = 'rgba(0,0,0,0)',
                            legend = dict(x = 0, y = -0.3, orientation = 'h'),
                            width = 500,
                            height = 500,
                            margin = dict(l=0, r=20, t=40, b=10),
                            autosize = False
                            )
        fig.update_xaxes(range = [0.02, 0.3], showline = True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(range = [-.065, .04], showline = True, linewidth=2, linecolor='black', mirror = True)

        figw = go.FigureWidget(fig)
        return figw

    def intercept_plot(self):
        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        black_line = "rgba(0,0,0,0.2)"
        models = sorted(list(self.worstcase_intercept.keys()),reverse = True)
        fig = go.Figure()
        for i, params in enumerate(models):
            alpha, kappa = params
            wc_alpha, wc_kappa = self.worstcase_intercept_path[0][i], self.worstcase_intercept_path[1][i]
            if kappa == 0.169 and alpha == 0.0:
                fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = True, name = r"$\text{{baseline: }}(\hat{{\alpha}}, \hat{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa),
                marker = dict(size = 8), showlegend = True, legendgroup = 'Baseline Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = np.array(self.worstcase_intercept_path)[1,:], y = np.array(self.worstcase_intercept_path)[0,:], visible = True, line = dict(dash = 'dashdot'), name = "worst-case exp path", showlegend = True, legendgroup = 'exp path'))
                fig.add_trace(go.Scatter(x = [None], y = [None], visible = False, marker = dict(size = 8), name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), 
                showlegend = True, legendgroup = 'Worrisome Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = [None], y = [None], visible = False, line = dict(color = blue_line), 
                    fill = 'toself', mode = 'lines', fillcolor = blue_fill, showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))
                fig.add_trace(go.Scatter(x = [None], y = [None], visible = False, marker = dict(size = 8), name = r"$\text{{worstcase: }}(\alpha, \kappa) = ({:.3f},{:.3f})$".format(wc_alpha, wc_kappa), 
                    showlegend = True, legendgroup = 'Worstcase Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = self.isos_intercept[wc_alpha, wc_kappa][:,0], y = self.isos_intercept[wc_alpha, wc_kappa][:,1], visible = False, legendgroup = 'iso', name = "iso value = {:.2f}".format(self.isos_intercept_val[wc_alpha, wc_kappa]), showlegend = True, line = dict(color = black_line)))

            else:
                fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = False, marker = dict(size = 8), name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa),
                    showlegend = True, legendgroup = 'Worrisome Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = self.worstcase_intercept[alpha, kappa][:,0], y = self.worstcase_intercept[alpha, kappa][:,1], visible = False, line = dict(color = blue_line), 
                    fill = 'toself', mode = 'lines', fillcolor = blue_fill, showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))
                fig.add_trace(go.Scatter(x = [wc_kappa], y = [wc_alpha], visible = False, marker = dict(size = 8), name = r"$\text{{worstcase: }}(\alpha, \kappa) = ({:.3f},{:.3f})$".format(wc_alpha, wc_kappa), 
                    showlegend = True, legendgroup = 'Worstcase Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = self.isos_intercept[wc_alpha, wc_kappa][:,0], y = self.isos_intercept[wc_alpha, wc_kappa][:,1], visible = False,  legendgroup = 'iso', name = "iso value = {:.2f}".format(self.isos_intercept_val[wc_alpha, wc_kappa]), showlegend = True, line = dict(color = black_line)))
                    
        fig.add_trace(go.Scatter(x = np.array(models)[:,1], y = np.array(models)[:,0], visible = True, line = dict(color = "rgba(0,0,0,0.3)", dash = 'dashdot'), name = "worrisome exp path", showlegend = False, legendgroup = 'worrisome exp path'))
        xs = np.linspace(0.02, 0.3, 50)
        ys = np.linspace(-0.065, 0.04, 50)
        fig.add_trace(go.Scatter(x = [0.169] * 50, y = ys, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))
        fig.add_trace(go.Scatter(x = xs, y = [0] * 50, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))

        steps = []
        fig.data[10 * 4 + 4]['visible'] = True
        fig.data[10 * 4 + 5]['visible'] = True
        fig.data[10 * 4 + 2]['visible'] = True
        fig.data[10 * 4 + 3]['visible'] = True

        for i in range(len(models)):
            if i == 0:
                label = '{:.3f}'.format(models[i][0])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][1] = True
                step['args'][1][2] = True
                step['args'][1][3] = True
                step['args'][1][4] = True
                step['args'][1][5] = True

            else:
                label = '{:.3f}'.format(models[i][0])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][1] = True
                step['args'][1][i*4 + 4] = True
                step['args'][1][i*4 + 5] = True
                step['args'][1][i*4 + 2] = True
                step['args'][1][i*4 + 3] = True

            step['args'][1][-1] = True
            step['args'][1][-2] = True
            step['args'][1][-3] = True
            steps.append(step)

        sliders = [dict(active = 10,
                    currentvalue = {"prefix": 'alpha: '},
                    pad = {"t": len(models)},
                    steps = steps,
                    x = 0,
                    y = 0.35)]

        fig.update_layout(title = "concern about intercept", titlefont = dict(size = 20), 
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text=r'$\kappa$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text=r'$\alpha$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            sliders = sliders,
                            plot_bgcolor = 'rgba(0,0,0,0)',
                            legend = dict(x = 0, y = -0.3, orientation = 'h'),
                            width = 500,
                            height = 550,
                            margin = dict(l=0, r=20, t=40, b=10),
                            autosize = False
                            )
        fig.update_xaxes(range = [0.05, 0.3], showline = True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(range = [-.06, .04], showline = True, linewidth=2, linecolor='black', mirror = True)

        figw = go.FigureWidget(fig)
        return figw

    def persistence_plot(self):
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        black_line = "rgba(0,0,0,0.2)"
        models = sorted(list(self.worstcase_persistence.keys()),reverse = True)
        fig = go.Figure()
        for i, params in enumerate(models):
            alpha, kappa = params
            wc_alpha, wc_kappa = self.worstcase_persistence_path[0][i], self.worstcase_persistence_path[1][i]

            if kappa == 0.169 and alpha == 0.0:
                fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = True, marker = dict(size = 8), name = r"$\text{{baseline: }}(\hat{{\alpha}}, \hat{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa),
                    showlegend = True, legendgroup = 'Baseline Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = np.array(self.worstcase_persistence_path)[1,:], y = np.array(self.worstcase_persistence_path)[0,:], visible = True, line = dict(dash = 'dashdot'), name = "worst-case exp path", showlegend = True, legendgroup = 'exp path'))
                fig.add_trace(go.Scatter(x = [None], y = [None], visible = False, marker = dict(size = 8), name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), 
                showlegend = True, legendgroup = 'Worrisome Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = [None], y = [None], visible = False, line = dict(color = black_line), 
                    fill = 'toself', mode = 'lines', fillcolor = black_fill, showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))
                fig.add_trace(go.Scatter(x = [None], y = [None], visible = False, marker = dict(size = 8), name = r"$\text{{worstcase: }}(\alpha, \kappa) = ({:.3f},{:.3f})$".format(wc_alpha, wc_kappa), 
                    showlegend = True, legendgroup = 'Worstcase Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = self.isos_persistence[wc_alpha, wc_kappa][:,0], y = self.isos_persistence[wc_alpha, wc_kappa][:,1], visible = False, legendgroup = 'iso', name = "iso value = {:.2f}".format(self.isos_persistence_val[wc_alpha, wc_kappa]), showlegend = True, line = dict(color = black_line)))

            else:
                fig.add_trace(go.Scatter(x = [kappa], y = [alpha], visible = False, marker = dict(size = 8), name = r"$\text{{worrisome: }}(\tilde{{\alpha}}, \tilde{{\kappa}}) = ({:.3f},{:.3f})$".format(alpha, kappa), 
                showlegend = True, legendgroup = 'Worrisome Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = self.worstcase_persistence[alpha, kappa][:,0], y = self.worstcase_persistence[alpha, kappa][:,1], visible = False, line = dict(color = black_line), 
                    fill = 'toself', mode = 'lines', fillcolor = black_fill, showlegend = True, name = r'$\text{boundary iso-}\varrho\text{ curve}$', legendgroup = 'isocurve'))
                fig.add_trace(go.Scatter(x = [wc_kappa], y = [wc_alpha], visible = False, marker = dict(size = 8), name = r"$\text{{worstcase: }}(\alpha, \kappa) = ({:.3f},{:.3f})$".format(wc_alpha, wc_kappa), 
                    showlegend = True, legendgroup = 'Worstcase Model', mode = 'markers'))
                fig.add_trace(go.Scatter(x = self.isos_persistence[wc_alpha, wc_kappa][:,0], y = self.isos_persistence[wc_alpha, wc_kappa][:,1], visible = False,  legendgroup = 'iso', name = "iso value = {:.2f}".format(self.isos_persistence_val[wc_alpha, wc_kappa]), showlegend = True, line = dict(color = black_line)))
        
        fig.add_trace(go.Scatter(x = np.array(models)[:,1], y = np.array(models)[:,0], visible = True, line = dict(color = "rgba(0,0,0,0.3)", dash = 'dashdot'), name = "worrisome exp path", showlegend = False, legendgroup = 'worrisome exp path'))
        xs = np.linspace(0.02, 0.3, 50)
        ys = np.linspace(-0.065, 0.04, 50)
        fig.add_trace(go.Scatter(x = [0.169] * 50, y = ys, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))
        fig.add_trace(go.Scatter(x = xs, y = [0] * 50, visible = True, line=dict(color='black', width = 1, dash = 'dot'), showlegend = False))

        steps = []
        fig.data[10 * 4 + 5]['visible'] = True
        fig.data[10 * 4 + 4]['visible'] = True
        fig.data[10 * 4 + 2]['visible'] = True
        fig.data[10 * 4 + 3]['visible'] = True

        for i in range(len(models)):
            if i == 0:
                label = '{:.3f}'.format(models[i][0])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][1] = True
                step['args'][1][2] = True
                step['args'][1][3] = True
                step['args'][1][4] = True
                step['args'][1][5] = True

            else:
                label = '{:.3f}'.format(models[i][0])
                step = dict(
                    method = 'restyle',
                    args = ['visible', [False] * len(fig.data)],
                    label = label
                )
                step['args'][1][0] = True
                step['args'][1][1] = True
                step['args'][1][i*4 + 4] = True
                step['args'][1][i*4 + 5] = True
                step['args'][1][i*4 + 2] = True
                step['args'][1][i*4 + 3] = True

            step['args'][1][-1] = True
            step['args'][1][-2] = True
            step['args'][1][-3] = True

            steps.append(step)

        sliders = [dict(active = 10,
                    currentvalue = {"prefix": 'alpha: '},
                    pad = {"t": len(models)},
                    steps = steps,
                    x = 0,
                    y = 0.35)]

        fig.update_layout(title = "concern about persistence", titlefont = dict(size = 20), 
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text=r'$\kappa$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text=r'$\alpha$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            sliders = sliders,
                            plot_bgcolor = 'rgba(0,0,0,0)',
                            legend = dict(x = 0, y = -0.3, orientation = 'h'),
                            width = 500,
                            height = 550,
                            margin = dict(l=0, r=20, t=40, b=10),
                            autosize = False
                            )
        fig.update_xaxes(range = [0.05, 0.3], showline = True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(range = [-.06, .04], showline = True, linewidth=2, linecolor='black', mirror = True)

        figw = go.FigureWidget(fig)
        return figw

    def ex_post_plot(self):

        fig = make_subplots(subplot_titles = ("investment ratio of the first capital", "stationary distribution of the states"),
            rows=2, cols=3,
            specs = [[{"rowspan":2}, {}, {}],
                    [None, {}, {}]],
            column_widths = [0.4, 0.4,0.1], row_heights = [0.2, 0.8],vertical_spacing=0.02, horizontal_spacing=0.07)

        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        show_lgd = True
        plotdata = self.ex_post
        for kappa_hat in self.kappa_hat_list:
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withoutr'], name="investment ratio w/o robustness control", 
                        line = dict(color = "rgba(0,0,0, 1)", width = 2),legendgroup = 'w/o robustness', showlegend = True, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withoutr_ld'], name="w/o robustness control lower bound", 
                        line = dict(color = black_line, width = 2, dash = 'dash'), legendgroup = 'w/o robustness', showlegend = False, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withoutr_ud'], name="w/o robustness control upper bound",
                        line = dict(color = black_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = black_fill, legendgroup = 'w/o robustness', visible = show_lgd,
                                    showlegend = False), row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withr'], name="investment ratio w/ robustness control", 
                        line = dict(color = "rgba(214,39,40, 1)", width = 2),legendgroup = 'w/ robustness', showlegend = True, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withr_ld'], name="w/ robustness control lower bound", 
                        line = dict(color = red_line, width = 2, dash = 'dash'), legendgroup = 'w/ robustness', showlegend = False, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withr_ud'], name="w/ robustness control upper bound",
                        line = dict(color = red_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = red_fill, legendgroup = 'w/ robustness', visible = show_lgd,
                                    showlegend = False), row=1, col=1)

            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['withoutr'][:,0], y=plotdata[kappa_hat]['density']['withoutr'][:,1], 
                                    legendgroup = 'w/o robustness (r)', showlegend = True, visible = show_lgd,
                                    name="stationary distribution w/o robustness control", line = dict(color = black_line)), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['withoutr_lb'][:,0], y=plotdata[kappa_hat]['density']['withoutr_lb'][:,1], 
                                    legendgroup = 'w/o robustness (r)', showlegend = False, visible = show_lgd,
                                    name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

            x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['withoutr_ub'][:,0], plotdata[kappa_hat]['density']['withoutr_ub'][:,1])

            fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                    legendgroup = 'w/o robustness (r)', showlegend = False,  visible = show_lgd,
                                    name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)
            fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd,
                                    legendgroup = 'w/o robustness (r)', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = black_fill,
                                    name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_base'][:,0], y=plotdata[kappa_hat]['density']['r_base'][:,1], 
                                    legendgroup = 'baseline w/ robustness', showlegend = True, visible = show_lgd,
                                    name="stationary distribution w/ robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_base_lb'][:,0], y=plotdata[kappa_hat]['density']['r_base_lb'][:,1],
                                    legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_base_ub'][:,0], y=plotdata[kappa_hat]['density']['r_base_ub'][:,1],
                                    legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

            x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['r_base_ub'][:,0], plotdata[kappa_hat]['density']['r_base_ub'][:,1])

            fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                    legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)
            fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd,
                                    legendgroup = 'baseline w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = red_fill,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)


            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst'][:,0], y=plotdata[kappa_hat]['density']['r_worst'][:,1],  visible = show_lgd,
                                    legendgroup = 'worstcase w/ robustness', showlegend = True,
                                    name="stationary distribution w/ robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst_lb'][:,0], y=plotdata[kappa_hat]['density']['r_worst_lb'][:,1], 
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst_ub'][:,0], y=plotdata[kappa_hat]['density']['r_worst_ub'][:,1],
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)

            x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['r_worst_ub'][:,0], plotdata[kappa_hat]['density']['r_worst_ub'][:,1])

            fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)
            fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd,
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = blue_fill,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)

            fig.add_trace(go.Scatter( x = [0, 1],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = True,
                            name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'), visible = show_lgd),
                            row = 2, col = 2)

            fig.add_trace(go.Scatter( x = [0, 0.025],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = False,
                            name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'), visible = show_lgd),
                            row = 2, col = 3)

            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['right']['f_base'], y=plotdata[kappa_hat]['right']['x'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = red_fill, line = dict(color = red_line, dash = 'dash', width = 1)), row=2, col=3)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['right']['f_without'], y=plotdata[kappa_hat]['right']['x'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=2, col=3)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['right']['f_worst'], y=plotdata[kappa_hat]['right']['x'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=2, col=3)


            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_without'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=1, col=2)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_base'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = red_fill, line = dict(color = red_line, width = 1, dash = 'dash')), row=1, col=2)
            fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_worst'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=1, col=2)
            if show_lgd == True:
                show_lgd = False

        steps = []
        for i in range(len(self.kappa_hat_list)):

            label = '{:.3f}'.format(self.kappa_hat_list[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            for j in range(28):
                step['args'][1][j + i * 28] = True

            # step['args'][1][l:] = [True] * len(self.isos)
            steps.append(step)

        sliders = [dict(active = 0,
                    currentvalue = {"prefix": 'kappa_hat: '},
                    pad = {"t": len(self.kappa_hat_list)},
                    steps = steps,
                    x = 0,
                    y = -0.05)]

        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_yaxes( showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_xaxes(showticklabels = False, row = 1, col = 2)
        fig.update_yaxes(showticklabels = False, row = 1, col = 2)
        fig.update_xaxes(range = [0, 0.025], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(range = [-0.8, 0.8], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(title_text = r'$Z$', title_standoff = 0, range = [-0.8, 0.8], row = 2, col = 2)
        fig.update_xaxes(range = [0, 1], title_text = r'$R$',title_standoff = 0, row = 2, col = 2)
        fig.update_xaxes(title_text = r'$R$',title_standoff = 0, row = 1, col = 1)
        fig.update_yaxes(title_text = r'$d^*_1$', title_standoff = 0, range = [0.031, 0.035], row = 1, col = 1)

        fig.update_layout(height=800, width=950, plot_bgcolor = 'rgba(0,0,0,0)', legend = dict(x = 0, y = -0.25, orientation = 'h'), 
                          sliders = sliders, margin = dict(l=20, r=20, t=20, b=20),)
        fig.show()

    def ex_ante_plot(self):
    
        fig = make_subplots(subplot_titles = ("investment ratio of the first capital", "stationary distribution of the states"),
            rows=2, cols=3,
            specs = [[{"rowspan":2}, {}, {}],
                    [None, {}, {}]],
            column_widths = [0.4, 0.4,0.1], row_heights = [0.2, 0.8],vertical_spacing=0.02, horizontal_spacing=0.07)

        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        show_lgd = True
        plotdata = self.ex_ante
        for beta_hat in self.beta_hat_list:
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withoutr'], name="investment ratio w/o robustness control", 
                        line = dict(color = "rgba(0,0,0, 1)", width = 2),legendgroup = 'w/o robustness', showlegend = True, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withoutr_ld'], name="w/o robustness control lower bound", 
                        line = dict(color = black_line, width = 2, dash = 'dash'), legendgroup = 'w/o robustness', showlegend = False, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withoutr_ud'], name="w/o robustness control upper bound",
                        line = dict(color = black_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = black_fill, legendgroup = 'w/o robustness', visible = show_lgd,
                                    showlegend = False), row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withr'], name="investment ratio w/ robustness control", 
                        line = dict(color = "rgba(214,39,40, 1)", width = 2),legendgroup = 'w/ robustness', showlegend = True, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withr_ld'], name="w/ robustness control lower bound", 
                        line = dict(color = red_line, width = 2, dash = 'dash'), legendgroup = 'w/ robustness', showlegend = False, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withr_ud'], name="w/ robustness control upper bound",
                        line = dict(color = red_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = red_fill, legendgroup = 'w/ robustness', visible = show_lgd,
                                    showlegend = False), row=1, col=1)

            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['withoutr'][:,0], y=plotdata[beta_hat]['density']['withoutr'][:,1], 
                                    legendgroup = 'w/o robustness (r)', showlegend = True, visible = show_lgd,
                                    name="stationary distribution w/o robustness control", line = dict(color = black_line)), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['withoutr_lb'][:,0], y=plotdata[beta_hat]['density']['withoutr_lb'][:,1], 
                                    legendgroup = 'w/o robustness (r)', showlegend = False, visible = show_lgd,
                                    name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

            x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['withoutr_ub'][:,0], plotdata[beta_hat]['density']['withoutr_ub'][:,1])

            fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                    legendgroup = 'w/o robustness (r)', showlegend = False,  visible = show_lgd,
                                    name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)
            fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd,
                                    legendgroup = 'w/o robustness (r)', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = black_fill,
                                    name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_base'][:,0], y=plotdata[beta_hat]['density']['r_base'][:,1], 
                                    legendgroup = 'baseline w/ robustness', showlegend = True, visible = show_lgd,
                                    name="stationary distribution w/ robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_base_lb'][:,0], y=plotdata[beta_hat]['density']['r_base_lb'][:,1],
                                    legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_base_ub'][:,0], y=plotdata[beta_hat]['density']['r_base_ub'][:,1],
                                    legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

            x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['r_base_ub'][:,0], plotdata[beta_hat]['density']['r_base_ub'][:,1])

            fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                    legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)
            fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd,
                                    legendgroup = 'baseline w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = red_fill,
                                    name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)


            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_worst'][:,0], y=plotdata[beta_hat]['density']['r_worst'][:,1],  visible = show_lgd,
                                    legendgroup = 'worstcase w/ robustness', showlegend = True,
                                    name="stationary distribution w/ robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_worst_lb'][:,0], y=plotdata[beta_hat]['density']['r_worst_lb'][:,1], 
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_worst_ub'][:,0], y=plotdata[beta_hat]['density']['r_worst_ub'][:,1],
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)

            x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['r_worst_ub'][:,0], plotdata[beta_hat]['density']['r_worst_ub'][:,1])

            fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)
            fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd,
                                    legendgroup = 'worstcase w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = blue_fill,
                                    name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)

            fig.add_trace(go.Scatter( x = [0, 1],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = True,
                            name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'), visible = show_lgd),
                            row = 2, col = 2)


            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['right']['f_base'], y=plotdata[beta_hat]['right']['x'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = red_fill, line = dict(color = red_line, dash = 'dash', width = 1)), row=2, col=3)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['right']['f_without'], y=plotdata[beta_hat]['right']['x'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=2, col=3)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['right']['f_worst'], y=plotdata[beta_hat]['right']['x'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=2, col=3)
            fig.add_trace(go.Scatter( x = [0, 0.025],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = False, visible = show_lgd,
                            name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'),),
                            row = 2, col = 3)


            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_without'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=1, col=2)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_base'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = red_fill, line = dict(color = red_line, width = 1, dash = 'dash')), row=1, col=2)
            fig.add_trace(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_worst'], fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                    fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=1, col=2)
            if show_lgd == True:
                show_lgd = False

        steps = []
        for i in range(len(self.beta_hat_list)):

            label = '{:.3f}'.format(self.beta_hat_list[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            for j in range(28):
                step['args'][1][j + i * 28] = True

            # step['args'][1][l:] = [True] * len(self.isos)
            steps.append(step)

        sliders = [dict(active = 0,
                    currentvalue = {"prefix": 'beta_hat: '},
                    pad = {"t": len(self.beta_hat_list)},
                    steps = steps,
                    x = 0,
                    y = -0.05)]

        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_yaxes( showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_xaxes(showticklabels = False, row = 1, col = 2)
        fig.update_yaxes(showticklabels = False, row = 1, col = 2)
        fig.update_xaxes(range = [0, 0.025], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(range = [-0.8, 0.8], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(title_text = r'$Z$', title_standoff = 0, range = [-0.8, 0.8], row = 2, col = 2)
        fig.update_xaxes(range = [0, 1], title_text = r'$R$',title_standoff = 0, row = 2, col = 2)
        fig.update_xaxes(title_text = r'$R$',title_standoff = 0, row = 1, col = 1)
        fig.update_yaxes(title_text = r'$d^*_1$', title_standoff = 0, range = [0.031, 0.035], row = 1, col = 1)

        fig.update_layout(height=800, width=950, plot_bgcolor = 'rgba(0,0,0,0)', legend = dict(x = 0, y = -0.25, orientation = 'h'),
                          sliders = sliders, margin = dict(l=20, r=20, t=20, b=20),)
        fig.show()

    def ex_ante_animation(self):
        
        fig = make_subplots(subplot_titles = ("investment ratio of the first capital", "stationary distribution of the states"),
            rows=2, cols=3,
            specs = [[{"rowspan":2}, {}, {}],
                    [None, {}, {}]],
            column_widths = [0.4, 0.4,0.1], row_heights = [0.2, 0.8],vertical_spacing=0.02, horizontal_spacing=0.07)

        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        show_lgd = True
        plotdata = self.ex_ante

        beta_hat = self.beta_hat_list[0]

        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withoutr'], name="investment ratio w/o robustness control", hoverinfo= 'name',
 
                    line = dict(color = "rgba(0,0,0, 1)", width = 2),legendgroup = 'w/o robustness', showlegend = True, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withoutr_ld'], name="w/o robustness control lower bound", hoverinfo= 'name',
                    line = dict(color = black_line, width = 2, dash = 'dash'), legendgroup = 'w/o robustness', showlegend = False, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withoutr_ud'], name="w/o robustness control upper bound",hoverinfo= 'name',
                    line = dict(color = black_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = black_fill, legendgroup = 'w/o robustness', visible = show_lgd,
                                showlegend = False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withr'], name="investment ratio w/ robustness control", hoverinfo= 'name',
                    line = dict(color = "rgba(214,39,40, 1)", width = 2),legendgroup = 'w/ robustness', showlegend = True, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withr_ld'], name="w/ robustness control lower bound", hoverinfo= 'name',
                    line = dict(color = red_line, width = 2, dash = 'dash'), legendgroup = 'w/ robustness', showlegend = False, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['left']['x'], y=plotdata[beta_hat]['left']['withr_ud'], name="w/ robustness control upper bound", hoverinfo= 'name',
                    line = dict(color = red_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = red_fill, legendgroup = 'w/ robustness', visible = show_lgd,
                                showlegend = False), row=1, col=1)

        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['withoutr'][:,0], y=plotdata[beta_hat]['density']['withoutr'][:,1], hoverinfo= 'name',
                                legendgroup = 'w/o robustness (r)', showlegend = True, visible = show_lgd,
                                name="stationary distribution w/o robustness control", line = dict(color = black_line)), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['withoutr_lb'][:,0], y=plotdata[beta_hat]['density']['withoutr_lb'][:,1], hoverinfo= 'name',
                                legendgroup = 'w/o robustness (r)', showlegend = False, visible = show_lgd,
                                name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

        x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['withoutr_ub'][:,0], plotdata[beta_hat]['density']['withoutr_ub'][:,1])

        fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                legendgroup = 'w/o robustness (r)', showlegend = False,  visible = show_lgd, hoverinfo= 'name',
                                name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)
        fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'w/o robustness (r)', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = black_fill,
                                name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_base'][:,0], y=plotdata[beta_hat]['density']['r_base'][:,1], hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = True, visible = show_lgd,
                                name="stationary distribution w/ robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_base_lb'][:,0], y=plotdata[beta_hat]['density']['r_base_lb'][:,1], hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_base_ub'][:,0], y=plotdata[beta_hat]['density']['r_base_ub'][:,1], hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

        x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['r_base_ub'][:,0], plotdata[beta_hat]['density']['r_base_ub'][:,1])

        fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd, hoverinfo= 'name',
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)
        fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = red_fill,
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)


        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_worst'][:,0], y=plotdata[beta_hat]['density']['r_worst'][:,1],  visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = True,
                                name="stationary distribution w/ robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_worst_lb'][:,0], y=plotdata[beta_hat]['density']['r_worst_lb'][:,1],  hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['density']['r_worst_ub'][:,0], y=plotdata[beta_hat]['density']['r_worst_ub'][:,1], hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)

        x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['r_worst_ub'][:,0], plotdata[beta_hat]['density']['r_worst_ub'][:,1])

        fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd, hoverinfo= 'name',
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)
        fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = blue_fill,
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)

        fig.add_trace(go.Scatter( x = [0, 1],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = True, hoverinfo= 'name',
                        name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'), visible = show_lgd),
                        row = 2, col = 2)


        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['right']['f_base'], y=plotdata[beta_hat]['right']['x'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = red_fill, line = dict(color = red_line, dash = 'dash', width = 1)), row=2, col=3)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['right']['f_without'], y=plotdata[beta_hat]['right']['x'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=2, col=3)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['right']['f_worst'], y=plotdata[beta_hat]['right']['x'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=2, col=3)
        fig.add_trace(go.Scatter( x = [0, 0.025],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = False, hoverinfo= 'name', visible = show_lgd,
                        name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'),),
                        row = 2, col = 3)


        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_without'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=1, col=2)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_base'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = red_fill, line = dict(color = red_line, width = 1, dash = 'dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_worst'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=1, col=2)
            # if show_lgd == True:
            #     show_lgd = False
        
        frames = []
        for beta_hat in self.beta_hat_list:
            frame = {"data": [], "name": beta_hat, "traces": np.arange(0,28).tolist()}
            frame['data'].append(go.Scatter(y=plotdata[beta_hat]['left']['withoutr'],))

            frame['data'].append(go.Scatter( y=plotdata[beta_hat]['left']['withoutr_ld'],))
            frame['data'].append(go.Scatter(y=plotdata[beta_hat]['left']['withoutr_ud'],))
            frame['data'].append(go.Scatter( y=plotdata[beta_hat]['left']['withr'],))
            frame['data'].append(go.Scatter(y=plotdata[beta_hat]['left']['withr_ld'],))
            frame['data'].append(go.Scatter( y=plotdata[beta_hat]['left']['withr_ud'], ))

            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['withoutr'][:,0], y=plotdata[beta_hat]['density']['withoutr'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['withoutr_lb'][:,0], y=plotdata[beta_hat]['density']['withoutr_lb'][:,1],))

            x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['withoutr_ub'][:,0], plotdata[beta_hat]['density']['withoutr_ub'][:,1])

            frame['data'].append(go.Scatter(x = x_down, y = y_down,))
            frame['data'].append(go.Scatter(x = x_up, y = y_up,))

            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['r_base'][:,0], y=plotdata[beta_hat]['density']['r_base'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['r_base_lb'][:,0], y=plotdata[beta_hat]['density']['r_base_lb'][:,1],))

            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['r_base_ub'][:,0], y=plotdata[beta_hat]['density']['r_base_ub'][:,1],))

            x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['r_base_ub'][:,0], plotdata[beta_hat]['density']['r_base_ub'][:,1])

            frame['data'].append(go.Scatter(x = x_down, y = y_down,))
            frame['data'].append(go.Scatter(x = x_up, y = y_up,))


            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['r_worst'][:,0], y=plotdata[beta_hat]['density']['r_worst'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['r_worst_lb'][:,0], y=plotdata[beta_hat]['density']['r_worst_lb'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['density']['r_worst_ub'][:,0], y=plotdata[beta_hat]['density']['r_worst_ub'][:,1],))

            x_down, y_down, x_up, y_up = half_split(plotdata[beta_hat]['density']['r_worst_ub'][:,0], plotdata[beta_hat]['density']['r_worst_ub'][:,1])

            frame['data'].append(go.Scatter(x = x_down, y = y_down,))
            frame['data'].append(go.Scatter(x = x_up, y = y_up,))

            frame['data'].append(go.Scatter( x = [0, 1],  y = [-.005/.017,] * 2, ))


            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['right']['f_base'], y=plotdata[beta_hat]['right']['x'], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['right']['f_without'], y=plotdata[beta_hat]['right']['x'], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['right']['f_worst'], y=plotdata[beta_hat]['right']['x'], ))
            frame['data'].append(go.Scatter( x = [0, 0.025],  y = [-.005/.017,] * 2,))


            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_without'], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_base'], ))
            frame['data'].append(go.Scatter(x=plotdata[beta_hat]['top']['x'], y=plotdata[beta_hat]['top']['f_worst'], ))

            frames.append(frame)

        updatemenus = [dict(type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[beta for beta in self.beta_hat_list], 
                                         dict(frame=dict(duration=500, redraw=False), 
                                              transition=dict(duration=0),
                                              easing='linear',
                                              fromcurrent=True,
                                              mode='immediate'
                                                                 )])],
                    direction= 'left', 
                    pad=dict(r= 10, t=85), 
                    showactive =True, x= 0, y= 0.1 , xanchor= 'right', yanchor= 'top')
            ]

        sliders = [{'yanchor': 'top',
            'xanchor': 'left', 
            'currentvalue': { 'prefix': 'beta_tilde: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 500.0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50}, 
            # 'len': 0.95, 
            'x': 0, 'y': 0, 
            'steps': [{'args': [[k], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                      'transition': {'duration': 0, 'easing': 'linear'}}], 
                       'label': k, 'method': 'animate'} for k in self.beta_hat_list       
                    ]}]
        fig.update(frames=frames),
        fig.update_layout(updatemenus=updatemenus,
                  sliders=sliders);

        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_yaxes( showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_xaxes(showticklabels = False, row = 1, col = 2)
        fig.update_yaxes(showticklabels = False, row = 1, col = 2)
        fig.update_xaxes(range = [0, 0.025], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(range = [-0.8, 0.8], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(title_text = r'$Z$', title_standoff = 0, range = [-0.8, 0.8], row = 2, col = 2)
        fig.update_xaxes(range = [0, 1], title_text = r'$R$',title_standoff = 0, row = 2, col = 2)
        fig.update_xaxes(title_text = r'$R$',title_standoff = 0, row = 1, col = 1)
        fig.update_yaxes(title_text = r'$d^*_1$', title_standoff = 0, range = [0.031, 0.035], row = 1, col = 1)

        fig.update_layout(height=800, width=950, plot_bgcolor = 'rgba(0,0,0,0)', legend = dict(x = 0, y = -0.25, orientation = 'h'),
                         margin = dict(l=20, r=20, t=20, b=20),)
        fig.show()

    def ex_post_animation(self):
        
        fig = make_subplots(subplot_titles = ("investment ratio of the first capital", "stationary distribution of the states"),
            rows=2, cols=3,
            specs = [[{"rowspan":2}, {}, {}],
                    [None, {}, {}]],
            column_widths = [0.4, 0.4,0.1], row_heights = [0.2, 0.8],vertical_spacing=0.02, horizontal_spacing=0.07)

        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        show_lgd = True
        plotdata = self.ex_post

        kappa_hat = self.kappa_hat_list[0]

        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withoutr'], name="investment ratio w/o robustness control", hoverinfo= 'name',
 
                    line = dict(color = "rgba(0,0,0, 1)", width = 2),legendgroup = 'w/o robustness', showlegend = True, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withoutr_ld'], name="w/o robustness control lower bound", hoverinfo= 'name',
                    line = dict(color = black_line, width = 2, dash = 'dash'), legendgroup = 'w/o robustness', showlegend = False, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withoutr_ud'], name="w/o robustness control upper bound",hoverinfo= 'name',
                    line = dict(color = black_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = black_fill, legendgroup = 'w/o robustness', visible = show_lgd,
                                showlegend = False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withr'], name="investment ratio w/ robustness control", hoverinfo= 'name',
                    line = dict(color = "rgba(214,39,40, 1)", width = 2),legendgroup = 'w/ robustness', showlegend = True, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withr_ld'], name="w/ robustness control lower bound", hoverinfo= 'name',
                    line = dict(color = red_line, width = 2, dash = 'dash'), legendgroup = 'w/ robustness', showlegend = False, visible = show_lgd),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['left']['x'], y=plotdata[kappa_hat]['left']['withr_ud'], name="w/ robustness control upper bound", hoverinfo= 'name',
                    line = dict(color = red_line, width = 2, dash = 'dash'), fill = 'tonexty', mode = 'lines', fillcolor = red_fill, legendgroup = 'w/ robustness', visible = show_lgd,
                                showlegend = False), row=1, col=1)

        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['withoutr'][:,0], y=plotdata[kappa_hat]['density']['withoutr'][:,1], hoverinfo= 'name',
                                legendgroup = 'w/o robustness (r)', showlegend = True, visible = show_lgd,
                                name="stationary distribution w/o robustness control", line = dict(color = black_line)), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['withoutr_lb'][:,0], y=plotdata[kappa_hat]['density']['withoutr_lb'][:,1], hoverinfo= 'name',
                                legendgroup = 'w/o robustness (r)', showlegend = False, visible = show_lgd,
                                name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

        x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['withoutr_ub'][:,0], plotdata[kappa_hat]['density']['withoutr_ub'][:,1])

        fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                legendgroup = 'w/o robustness (r)', showlegend = False,  visible = show_lgd, hoverinfo= 'name',
                                name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)
        fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'w/o robustness (r)', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = black_fill,
                                name="w/o robustness control", line = dict(color = black_line)), row=2, col=2)

        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_base'][:,0], y=plotdata[kappa_hat]['density']['r_base'][:,1], hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = True, visible = show_lgd,
                                name="stationary distribution w/ robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_base_lb'][:,0], y=plotdata[kappa_hat]['density']['r_base_lb'][:,1], hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_base_ub'][:,0], y=plotdata[kappa_hat]['density']['r_base_ub'][:,1], hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'dash')), row=2, col=2)

        x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['r_base_ub'][:,0], plotdata[kappa_hat]['density']['r_base_ub'][:,1])

        fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                legendgroup = 'baseline w/ robustness', showlegend = False, visible = show_lgd, hoverinfo= 'name',
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)
        fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'baseline w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = red_fill,
                                name="robustness control under baseline model", line = dict(color = red_line, dash = 'longdash', simplify = True)), row=2, col=2)


        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst'][:,0], y=plotdata[kappa_hat]['density']['r_worst'][:,1],  visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = True,
                                name="stationary distribution w/ robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst_lb'][:,0], y=plotdata[kappa_hat]['density']['r_worst_lb'][:,1],  hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst_ub'][:,0], y=plotdata[kappa_hat]['density']['r_worst_ub'][:,1], hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd,
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot')), row=2, col=2)

        x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['r_worst_ub'][:,0], plotdata[kappa_hat]['density']['r_worst_ub'][:,1])

        fig.add_trace(go.Scatter(x = x_down, y = y_down,
                                legendgroup = 'worstcase w/ robustness', showlegend = False, visible = show_lgd, hoverinfo= 'name',
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)
        fig.add_trace(go.Scatter(x = x_up, y = y_up, visible = show_lgd, hoverinfo= 'name',
                                legendgroup = 'worstcase w/ robustness', showlegend = False, fill = 'tonexty', mode = 'lines', fillcolor = blue_fill,
                                name="robustness control under worstcase model", line = dict(color = blue_line, dash = 'dot', simplify = True, shape = "spline")), row=2, col=2)

        fig.add_trace(go.Scatter( x = [0, 1],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = True, hoverinfo= 'name',
                        name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'), visible = show_lgd),
                        row = 2, col = 2)


        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['right']['f_base'], y=plotdata[kappa_hat]['right']['x'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = red_fill, line = dict(color = red_line, dash = 'dash', width = 1)), row=2, col=3)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['right']['f_without'], y=plotdata[kappa_hat]['right']['x'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=2, col=3)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['right']['f_worst'], y=plotdata[kappa_hat]['right']['x'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=2, col=3)
        fig.add_trace(go.Scatter( x = [0, 0.025],  y = [-.005/.017,] * 2, legendgroup = 'mean', showlegend = False, hoverinfo= 'name', visible = show_lgd,
                        name = "worstcase mean of Z in single capital economy", line = dict(color = 'rgba(44,160,44,0.6)', dash = 'dashdot'),),
                        row = 2, col = 3)


        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_without'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = black_fill, line = dict(color = black_line, width = 1)), row=1, col=2)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_base'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = red_fill, line = dict(color = red_line, width = 1, dash = 'dash')), row=1, col=2)
        fig.add_trace(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_worst'], hoverinfo= 'name', fill='tozeroy',mode='lines', showlegend = False, visible = show_lgd,
                                fillcolor = blue_fill, line = dict(color = blue_line, dash = 'dot', width = 1)), row=1, col=2)
            # if show_lgd == True:
            #     show_lgd = False
        
        frames = []
        for kappa_hat in self.kappa_hat_list:
            frame = {"data": [], "name": kappa_hat, "traces": np.arange(0,28).tolist()}
            frame['data'].append(go.Scatter(y=plotdata[kappa_hat]['left']['withoutr'],))

            frame['data'].append(go.Scatter( y=plotdata[kappa_hat]['left']['withoutr_ld'],))
            frame['data'].append(go.Scatter(y=plotdata[kappa_hat]['left']['withoutr_ud'],))
            frame['data'].append(go.Scatter( y=plotdata[kappa_hat]['left']['withr'],))
            frame['data'].append(go.Scatter(y=plotdata[kappa_hat]['left']['withr_ld'],))
            frame['data'].append(go.Scatter( y=plotdata[kappa_hat]['left']['withr_ud'], ))

            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['withoutr'][:,0], y=plotdata[kappa_hat]['density']['withoutr'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['withoutr_lb'][:,0], y=plotdata[kappa_hat]['density']['withoutr_lb'][:,1],))

            x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['withoutr_ub'][:,0], plotdata[kappa_hat]['density']['withoutr_ub'][:,1])

            frame['data'].append(go.Scatter(x = x_down, y = y_down,))
            frame['data'].append(go.Scatter(x = x_up, y = y_up,))

            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['r_base'][:,0], y=plotdata[kappa_hat]['density']['r_base'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['r_base_lb'][:,0], y=plotdata[kappa_hat]['density']['r_base_lb'][:,1],))

            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['r_base_ub'][:,0], y=plotdata[kappa_hat]['density']['r_base_ub'][:,1],))

            x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['r_base_ub'][:,0], plotdata[kappa_hat]['density']['r_base_ub'][:,1])

            frame['data'].append(go.Scatter(x = x_down, y = y_down,))
            frame['data'].append(go.Scatter(x = x_up, y = y_up,))


            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst'][:,0], y=plotdata[kappa_hat]['density']['r_worst'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst_lb'][:,0], y=plotdata[kappa_hat]['density']['r_worst_lb'][:,1], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['density']['r_worst_ub'][:,0], y=plotdata[kappa_hat]['density']['r_worst_ub'][:,1],))

            x_down, y_down, x_up, y_up = half_split(plotdata[kappa_hat]['density']['r_worst_ub'][:,0], plotdata[kappa_hat]['density']['r_worst_ub'][:,1])

            frame['data'].append(go.Scatter(x = x_down, y = y_down,))
            frame['data'].append(go.Scatter(x = x_up, y = y_up,))

            frame['data'].append(go.Scatter( x = [0, 1],  y = [-.005/.017,] * 2, ))


            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['right']['f_base'], y=plotdata[kappa_hat]['right']['x'], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['right']['f_without'], y=plotdata[kappa_hat]['right']['x'], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['right']['f_worst'], y=plotdata[kappa_hat]['right']['x'], ))
            frame['data'].append(go.Scatter( x = [0, 0.025],  y = [-.005/.017,] * 2,))


            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_without'], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_base'], ))
            frame['data'].append(go.Scatter(x=plotdata[kappa_hat]['top']['x'], y=plotdata[kappa_hat]['top']['f_worst'], ))

            frames.append(frame)

        updatemenus = [dict(type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[beta for beta in self.kappa_hat_list], 
                                         dict(frame=dict(duration=500, redraw=False), 
                                              transition=dict(duration=0),
                                              easing='linear',
                                              fromcurrent=True,
                                              mode='immediate'
                                                                 )])],
                    direction= 'left', 
                    pad=dict(r= 10, t=85), 
                    showactive =True, x= 0, y= 0.1 , xanchor= 'right', yanchor= 'top')
            ]

        sliders = [{'yanchor': 'top',
            'xanchor': 'left', 
            'currentvalue': { 'prefix': 'kappa_tilde: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 500.0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50}, 
            # 'len': 0.95, 
            'x': 0, 'y': 0, 
            'steps': [{'args': [[k], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                      'transition': {'duration': 0, 'easing': 'linear'}}], 
                       'label': k, 'method': 'animate'} for k in self.kappa_hat_list       
                    ]}]
        fig.update(frames=frames),
        fig.update_layout(updatemenus=updatemenus,
                  sliders=sliders);

        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_yaxes( showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_xaxes(showticklabels = False, row = 1, col = 2)
        fig.update_yaxes(showticklabels = False, row = 1, col = 2)
        fig.update_xaxes(range = [0, 0.025], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(range = [-0.8, 0.8], showticklabels = False, row = 2, col = 3)
        fig.update_yaxes(title_text = r'$Z$', title_standoff = 0, range = [-0.8, 0.8], row = 2, col = 2)
        fig.update_xaxes(range = [0, 1], title_text = r'$R$',title_standoff = 0, row = 2, col = 2)
        fig.update_xaxes(title_text = r'$R$',title_standoff = 0, row = 1, col = 1)
        fig.update_yaxes(title_text = r'$d^*_1$', title_standoff = 0, range = [0.031, 0.035], row = 1, col = 1)

        fig.update_layout(height=800, width=950, plot_bgcolor = 'rgba(0,0,0,0)', legend = dict(x = 0, y = -0.25, orientation = 'h'),
                         margin = dict(l=20, r=20, t=20, b=20),)
        fig.show()

    def twisting_plot(self):
        
        fig = go.Figure()

        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        red_line = "rgba(214,39,40, 0.6)"
        red_fill =  "rgba(214,39,40, 0.2)"
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        show_lgd = True
        plotdata = self.twistingfunction

        kappa_tilde = self.twisting_list[0]

        z_grid = np.linspace(-.8, .8, 100)
        show_lgd = True

        for kappa_tilde in self.twisting_list:
            xi_grid = np.zeros(100)
            xi2_grid = np.zeros(100)

            xi0_k, xi1_k, xi2_k = plotdata[kappa_tilde]['xi_k'] # model_asym_HSHS["xi0"],model_asym_HSHS["xi1"]       
            xi0_b, xi1_b, xi2_b = plotdata[kappa_tilde]['xi_b']   # model_asym_HSHS2["xi0"],model_asym_HSHS2["xi1"]     

            for i, z in enumerate(z_grid):
                xi_grid[i] = xi0_k + 2*xi1_k*z + xi2_k*z**2
                xi2_grid[i] = xi0_b + 2*xi1_b*z + xi2_b*z**2

            if show_lgd == True:
                fig.add_trace(go.Scatter(x = z_grid, y = np.ones(100) * 0.2 ** 2 * 0.5, name = r"$\text{target relative entropy }\mathsf{q}^2 / 2$", 
                        hoverinfo= 'name', line = dict(color = black_line, width = 2, dash = 'dot'), legendgroup = 'q', showlegend = True, visible = show_lgd))
            
            fig.add_trace(go.Scatter(x = z_grid, y = np.ones(100) * xi0_k, name = r"$\xi_0$", 
                    line = dict(color = black_line, width = 2, dash = 'dash'), legendgroup = 'const', showlegend = True, visible = show_lgd))

            fig.add_trace(go.Scatter(x = z_grid, y = xi_grid, name = r"$\xi^{{[\kappa]}}(z)\text{{ with worrisome }}(\tilde{{\alpha_z}}, \tilde{{\kappa}})=({:.3f}, {:.4f})$".format(plotdata[kappa_tilde]['xi_k_params'][1], plotdata[kappa_tilde]['xi_k_params'][2]),
                    line = dict(color = blue_line, width = 2), legendgroup = 'xi_kappa', showlegend = True, visible = show_lgd))
            fig.add_trace(go.Scatter(x = z_grid, y = xi2_grid, name = r"$\xi^{{[\beta]}}(z)\text{{ with worrisome }}(\tilde{{\alpha_z}}, \tilde{{\kappa}})=({:.3f}, {:.3f})$".format(plotdata[kappa_tilde]['xi_b_params'][1], plotdata[kappa_tilde]['xi_b_params'][3]),
                    line = dict(color = red_line, width = 2), legendgroup = 'xi_beta', showlegend = True, visible = show_lgd))

                
            if show_lgd == True:
                show_lgd = False
        
        steps = []
        for i in range(len(self.twisting_list)):
            
            label = '{:.4f}'.format(self.twisting_list[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            step['args'][1][0] = True
            step['args'][1][i*3 + 1] = True
            step['args'][1][i*3 + 2] = True
            step['args'][1][i*3 + 3] = True

            steps.append(step)

        sliders = [dict(active = 0,
                    currentvalue = {"prefix": 'kappa_tilde: '},
                    pad = {"t": 5},
                    steps = steps,

                    x = 0, y = 0.22)]



        fig.update_layout(sliders=sliders)
        
        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_yaxes( showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_xaxes(zeroline = True, zerolinewidth = 1, zerolinecolor='black')
        fig.update_yaxes(zeroline = True, zerolinewidth = 1, zerolinecolor='black')
        fig.update_xaxes(range = [-0.8, 0.8])
        fig.update_yaxes(range = [0, 0.025])
        fig.update_xaxes(title_text = r'$Z$', title_standoff = 0)

        fig.update_layout(height=500, width=950, plot_bgcolor = 'rgba(0,0,0,0)', legend = dict(x = 0, y = -0.25, orientation = 'h'),
                         margin = dict(l=20, r=20, t=20, b=20),)
        fig.show()

    def sym_r_irf_plot(self):
        blue_line = "rgba(31,119,178, 0.6)"
        red_line = "rgba(214,39,40, 0.6)"
        black_line = "rgba(0,0,0,0.6)"
        fig = go.Figure()
        xs = np.linspace(1,1000,1000)
        show_lgd = True
        for kappa in self.kappa_hat_list1:

            fig.add_trace(go.Scatter(x = xs, y = self.sym[kappa]['r_without'], visible = show_lgd, 
                line = dict(color = black_line), showlegend = True, name = 'w/o robustness',  legendgroup = 'w/o robustness',))
            fig.add_trace(go.Scatter(x = xs, y = self.sym[kappa]['r_base'], visible = show_lgd, 
                line = dict(color = red_line, dash = 'dash'), showlegend = True, name = "baseline w/ robustness",  legendgroup = 'baseline w/ robustness',))
            fig.add_trace(go.Scatter(x = xs, y = self.sym[kappa]['r_worst'], visible = show_lgd, 
                line = dict(color = blue_line, dash = 'dot'), showlegend = True, name = "worstcase w/ robustness",  legendgroup = 'worstcase w/ robustness',))
            if show_lgd == True:
                show_lgd = False

        fig.data[0]['visible'] = True
        fig.data[1]['visible'] = True
        fig.data[2]['visible'] = True

        steps = []
        for i in range(len(self.kappa_hat_list1)):
            
            label = '{}'.format(self.kappa_hat_list1[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            step['args'][1][i * 3] = True
            step['args'][1][i * 3 + 1] = True
            step['args'][1][i * 3 + 2] = True
            steps.append(step)

        sliders = [dict(active = 0,
                    currentvalue = {"prefix": 'kappa_tilde: '},
                    pad = {"t": 20},
                    steps = steps,
                    x = 0, y = 0)]

        fig.update_layout(title = dict(text = "Symmetric returns", font = dict(size = 20)),
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text='Horizon', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text=r'$R$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            sliders = sliders,
                            plot_bgcolor = 'rgba(0,0,0,0)',
                            legend = dict(x = 0, y = -0.3, orientation = 'h'),
                            width = 500,
                            height = 500,
                            margin = dict(l=20, r=10, t=40, b=10),
                            autosize = False
                            )
        fig.update_xaxes(range = [0, 1000], showline = True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(range = [0, .20], showline = True, linewidth=2, linecolor='black', mirror = True)
        # fig.show()
        figw = go.FigureWidget(fig)
        return figw

    def asym_r_irf_plot(self):
        blue_line = "rgba(31,119,178, 0.6)"
        red_line = "rgba(214,39,40, 0.6)"
        black_line = "rgba(0,0,0,0.6)"
        fig = go.Figure()
        xs = np.linspace(1,1000,1000)
        show_lgd = True
        for beta in self.beta_hat_list1:

            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['r_without'], visible = show_lgd, 
                line = dict(color = black_line), showlegend = True, name = 'w/o robustness',  legendgroup = 'w/o robustness',))
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['r_base'], visible = show_lgd, 
                line = dict(color = red_line, dash = 'dash'), showlegend = True, name = "baseline w/ robustness",  legendgroup = 'baseline w/ robustness',))
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['r_worst'], visible = show_lgd, 
                line = dict(color = blue_line, dash = 'dot'), showlegend = True, name = "worstcase w/ robustness",  legendgroup = 'worstcase w/ robustness',))
            if show_lgd == True:
                show_lgd = False

        fig.data[0]['visible'] = True
        fig.data[1]['visible'] = True
        fig.data[2]['visible'] = True

        steps = []
        for i in range(len(self.beta_hat_list1)):
            
            label = '{}'.format(self.beta_hat_list1[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            step['args'][1][i * 3] = True
            step['args'][1][i * 3 + 1] = True
            step['args'][1][i * 3 + 2] = True
            steps.append(step)

        sliders = [dict(active = 0,
                    currentvalue = {"prefix": 'beta_tilde: '},
                    pad = {"t": 20},
                    steps = steps,
                    x = 0, y = 0)]

        fig.update_layout(title = dict(text = "Asymmetric returns", font = dict(size = 20)),
                            xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(
                                                text='Horizon', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(
                                                text=r'$R$', font=dict(size=16)),
                                                    tickfont=dict(size=12), showgrid = False, title_standoff = 0),
                            sliders = sliders,
                            plot_bgcolor = 'rgba(0,0,0,0)',
                            legend = dict(x = 0, y = -0.3, orientation = 'h'),
                            width = 500,
                            height = 500,
                            margin = dict(l=20, r=10, t=40, b=10),
                            autosize = False
                            )
        fig.update_xaxes(range = [0, 1000], showline = True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(range = [0, .20], showline = True, linewidth=2, linecolor='black', mirror = True)
        # fig.show()
        figw = go.FigureWidget(fig)
        return figw

    def Z_plot(self):
        fig = make_subplots(subplot_titles = (r"$\mu_z(Z,R)$", "IRF of Z to idiosyncratic capital shock"),
            rows=1, cols=2,)
        blue_line = "rgba(31,119,178, 0.6)"
        blue_fill = "rgba(31,119,178, 0.2)"
        # red_line = "rgba(214,39,40, 0.6)"
        # red_fill =  "rgba(214,39,40, 0.2)"
        black_line = "rgba(0,0,0,0.6)"
        black_fill = "rgba(0,0,0, 0.2)"
        show_lgd = True
        xs = self.asym[0.5]['x']
        x2s = np.linspace(1,600,600)
        for beta in self.beta_hat_list1:
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['mu_z_base'], name=r"$\mu_z(Z,R)\text{: baseline}$", 
                        line = dict(color = black_line, width = 2), legendgroup = 'baseline', showlegend = True, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['mu_z_base_ud'], name="baseline lower bound", 
                        line = dict(color = black_line, width = 2), legendgroup = 'baseline', showlegend = False, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['mu_z_base_ld'], name="baseline upper bound",
                        line = dict(color = black_line, width = 2), fill = 'tonexty', mode = 'lines', fillcolor = black_fill, legendgroup = 'baseline', visible = show_lgd,
                                    showlegend = False), row=1, col=1)
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['mu_z_worst'], name = r"$\mu_z(Z,R)\text{: worst}$", 
                        line = dict(color = blue_line, width = 2, dash = 'dot'), legendgroup = 'worst', showlegend = True, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['mu_z_worst_ud'], name="worst lower bound", 
                        line = dict(color = blue_line, width = 2, dash = 'dot'), legendgroup = 'worst', showlegend = False, visible = show_lgd),
                        row=1, col=1)
            fig.add_trace(go.Scatter(x = xs, y = self.asym[beta]['mu_z_worst_ld'], name="worst upper bound",
                        line = dict(color = blue_line, width = 2, dash = 'dot'), fill = 'tonexty', mode = 'lines', fillcolor = blue_fill, legendgroup = 'worst', visible = show_lgd,
                                    showlegend = False), row=1, col=1)
            fig.add_trace(go.Scatter(x = np.ones(len(np.linspace(-.014, .0105))) * self.asym[beta]['z_median'], y = np.linspace(-.014, .0105), name="Median of Z",
                        line = dict(color = black_line, width = 2, dash = 'dashdot'),  legendgroup = 'median', visible = show_lgd,
                                    showlegend = False), row=1, col=1)
            fig.add_trace(go.Scatter(x = x2s, y = self.asym[beta]['z_base'], name="IRF: baseline", 
                        line = dict(color = black_line, width = 2), legendgroup = 'IRFbaseline', showlegend = True, visible = show_lgd),
                        row=1, col=2)
            fig.add_trace(go.Scatter(x = x2s, y = self.asym[beta]['z_worst'], name="IRF: worstcase", 
                        line = dict(color = blue_line, width = 2, dash = 'dot'), legendgroup = 'worstIRF', showlegend = True, visible = show_lgd),
                        row=1, col=2)
            if show_lgd == True:
                show_lgd = False

        steps = []
        for i in range(len(self.beta_hat_list)):

            label = '{:.2f}'.format(self.beta_hat_list1[i])
            step = dict(
                method = 'restyle',
                args = ['visible', [False] * len(fig.data)],
                label = label
            )
            for j in range(9):
                step['args'][1][j + i * 9] = True

            # step['args'][1][l:] = [True] * len(self.isos)
            steps.append(step)

        sliders = [dict(active = 0,
                    currentvalue = {"prefix": 'beta_tilde: '},
                    pad = {"t": len(self.beta_hat_list1)},
                    steps = steps,
                    x = 0,
                    y = 0.15)]

        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_yaxes( showline = True, linewidth=1, linecolor='black', mirror = True)
        fig.update_xaxes(zeroline = True, zerolinewidth = 1, zerolinecolor='black', row = 1, col = 2)
        fig.update_yaxes(zeroline = True, zerolinewidth = 1, zerolinecolor='black', row = 1, col = 1)
        fig.update_yaxes(zeroline = True, zerolinewidth = 1, zerolinecolor='black', row = 1, col = 2)
        fig.update_xaxes(range = [-.014, .0105], row = 1, col = 1)
        fig.update_yaxes(range = [-.014, .0105], row = 1, col = 1)
        fig.update_xaxes(range = [0, 1], title_text = r'$R$',title_standoff = 0, row = 1, col = 1)
        fig.update_xaxes(range = [0, 600], title_text = 'Horizon (Quarters)',title_standoff = 0, row = 1, col = 2)
        fig.update_yaxes(range = [-0.001, 0.01], row = 1, col = 2)

        fig.update_layout(height=500, width=950, plot_bgcolor = 'rgba(0,0,0,0)', legend = dict(x = 0, y = -0.25, orientation = 'h'),
                          sliders = sliders, margin = dict(l=20, r=20, t=40, b=40),)
        fig.show()
            
    def shockplot(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 4), sharex=True, sharey=True)

        ax[0].set_title("Shock to capital", fontsize=16)
        ax[0].plot(self.shock_densities['x11'], self.shock_densities['y11'], lw=2, color=colors[1], 
                label=r"Symmetric (concern about $\kappa$)")
        ax[0].fill_between(self.shock_densities['x11'], 0, self.shock_densities['y11'], color=colors[1], alpha=.15)
        ax[0].plot(self.shock_densities['x12'], self.shock_densities['y12'], lw=2, color=colors[0], 
                label=r"Symmetric returns with $\xi^{[\beta]}$")
        ax[0].plot(self.shock_densities['x13'], self.shock_densities['y13'], lw=2, color=colors[1], linestyle=':', 
                label=r'Asymmetric (concern about $\kappa$)')
        ax[0].fill_between(self.shock_densities['x13'], 0, self.shock_densities['y13'], color=colors[1], alpha=.1)
        ax[0].plot(self.shock_densities['x14'], self.shock_densities['y14'], lw=2, color=colors[0], linestyle=':',
                label=r'Asymmetric (concern about $\beta$)')
        ax[0].fill_between(self.shock_densities['x14'], 0, self.shock_densities['y14'], color=colors[0], alpha=.1)
        ax[0].fill_between(self.shock_densities['x12'], 0, self.shock_densities['y12'], color=colors[0], alpha=.15)
        ax[0].axhline(0, lw=1.5, color='k')
        ax[0].axvline(0, lw=1, color='k')

        ax[1].set_title("Shock to long run risk state", fontsize=16)
        ax[1].plot(self.shock_densities['x21'], self.shock_densities['y21'], lw=2, color=colors[1], 
                label=r"Symmetric with $\xi^{[\kappa]}$")
        ax[1].fill_between(self.shock_densities['x21'], 0, self.shock_densities['y21'], color=colors[1], alpha=.15)
        ax[1].plot(self.shock_densities['x22'], self.shock_densities['y22'], lw=2, color=colors[1], linestyle=':',
                label=r"Asymmetric with $\xi^{[\kappa]}$")
        ax[1].fill_between(self.shock_densities['x22'], 0, self.shock_densities['y22'], color=colors[1], alpha=.1)
        ax[1].plot(self.shock_densities['x23'], self.shock_densities['y23'], lw=2, color=colors[0], 
                label=r"Symmetric with $\xi^{[\beta]}$")
        ax[1].plot(self.shock_densities['x24'], self.shock_densities['y24'], lw=2, color=colors[0], linestyle=':',
                label=r"Asymmetric with $\xi^{[\beta]}$")
        ax[1].fill_between(self.shock_densities['x24'], 0, self.shock_densities['y24'], color=colors[0], alpha=.1)
        ax[1].fill_between(self.shock_densities['x23'], 0, self.shock_densities['y23'], color=colors[0], alpha=.15)

        ax[1].axhline(0, lw=1.5, color='k')
        ax[1].legend(loc=2, fontsize=12, ncol=2, frameon=True, framealpha=1.0)
        ax[1].set_xlim([-.12, .17])
        ax[1].axvline(0, lw=1, color='k')
        plt.tight_layout()

def irf_figure1(model, shock=0, dim='R', ylim_left=None, ylim_right=None):

    if dim=='R':
        dec1 = 1
        dec9 = 2
    elif dim=='Z':
        dec1 = 3
        dec9 = 4

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))

    ax[0].plot(100*model.R_irf[:, 2, shock, 0], lw=2, color='k', label='without robustness')
    ax[0].plot(100*model.R_irf[:, 2, shock, dec1], lw=2, color='k', linestyle='--', alpha=.8)
    ax[0].plot(100*model.R_irf[:, 2, shock, dec9], lw=2, color='k', linestyle=':', alpha=.8)
    ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 2, shock, dec1], 100*model.R_irf[:, 2, shock, 0],
                        color='k', alpha=.1)
    ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 2, shock, 0], 100*model.R_irf[:, 2, shock, dec9],
                        color='k', alpha=.1)

    ax[0].plot(100*model.R_irf[:, 0, shock, 0], lw=2, color=colors[3], label='robust control under baseline')
    ax[0].plot(100*model.R_irf[:, 0, shock, dec1], lw=2, color=colors[3], linestyle='--', alpha=.8)
    ax[0].plot(100*model.R_irf[:, 0, shock, dec9], lw=2, color=colors[3], linestyle=':', alpha=.8)
    ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 0, shock, dec1], 100*model.R_irf[:, 0, shock, 0],
                       color=colors[3], alpha=.1)
    ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 0, shock, 0], 100*model.R_irf[:, 0, shock, dec9],
                       color=colors[3], alpha=.1)
    ax[0].plot(100*model.R_irf[:, 1, shock, 0], lw=2, color=colors[0], label='robust control under worst-case')
    ax[0].plot(100*model.R_irf[:, 1, shock, dec1], lw=2, color=colors[0], linestyle='--', alpha=.8)
    ax[0].plot(100*model.R_irf[:, 1, shock, dec9], lw=2, color=colors[0], linestyle=':', alpha=.8)
    ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 1, shock, dec1], 100*model.R_irf[:, 1, shock, 0],
                       color=colors[0], alpha=.1)
    ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 1, shock, 0], 100*model.R_irf[:, 1, shock, dec9],
                       color=colors[0], alpha=.1)
    ax[0].axhline(0, color='k', lw=1)
    ax[0].set_title("IRF of R w.r.t. capital shock")
    if ylim_left:
        ax[0].set_ylim(ylim_left)
    ax[0].legend(loc='best')

    ax[1].plot(model.Z_irf[:, 0, shock, 0], lw=2, color=colors[3])
    ax[1].plot(model.Z_irf[:, 0, shock, dec1], lw=2, color=colors[3], linestyle='--', alpha=.8)
    ax[1].plot(model.Z_irf[:, 0, shock, dec9], lw=2, color=colors[3], linestyle=':', alpha=.8)

    ax[1].plot(model.Z_irf[:, 1, shock, 0], lw=2, color=colors[0])
    ax[1].plot(model.Z_irf[:, 1, shock, dec1], lw=2, color=colors[0], linestyle='--', alpha=.8)
    ax[1].plot(model.Z_irf[:, 1, shock, dec9], lw=2, color=colors[0], linestyle=':', alpha=.8)

    ax[1].plot(model.Z_irf[:, 2, shock, 0], lw=2, color='k')
    ax[1].plot(model.Z_irf[:, 2, shock, dec1], lw=2, color='k', linestyle='--', alpha=.8)
    ax[1].plot(model.Z_irf[:, 2, shock, dec9], lw=2, color='k', linestyle=':', alpha=.8)
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_title("IRF of Z w.r.t. capital shock")
    if ylim_right:
        ax[1].set_ylim(ylim_right)
    plt.tight_layout()

def fig_muZ(m1, m2):
    model = m1

    ind_ld_z, ind_med_z, ind_ud_z = model.ind_ld_z, model.ind_med_z, model.ind_ud_z
    ind_ld_r, ind_med_r, ind_ud_r = model.ind_ld_r, model.ind_med_r, model.ind_ud_r

    pii, rr, zz = model.pii, model.rr, model.zz
    mu_pii_noR, mu_pii, mu_pii_wc = model.mu_pii_noR, model.mu_pii, model.mu_pii_wc
    mu_z, mu_z_wc = model.mu_z, model.mu_z_wc

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

    inn_r = slice(model.ind_ld_r_noR, model.ind_ud_r_noR)
    ax[0].plot(pii[:, 0], mu_z[:, model.ind_med_z], color='k', lw=3, label="under baseline")
    ax[0].plot(pii[:, 0], mu_z[:, model.ind_ld_z], color='k', lw=2, linestyle='--', alpha=.4)
    ax[0].plot(pii[:, 0], mu_z[:, model.ind_ud_z], color='k', lw=2, linestyle=':', alpha=.4)
    ax[0].fill_between(pii[:, 0], mu_z[:, ind_ld_z], mu_z[:, ind_ud_z],
                                  color='k', alpha=.1, lw=3)

    ax[0].plot(pii[:, 0], mu_z_wc[:, ind_med_z], color=colors[0], alpha=.8, lw=3,
                     label="under worst-case")
    ax[0].plot(pii[:, 0], mu_z_wc[:, ind_ld_z], color=colors[0], linestyle='--', lw=2, alpha=.5)
    ax[0].plot(pii[:, 0], mu_z_wc[:, ind_ud_z], color=colors[0], linestyle=':', lw=2, alpha=.5)
    ax[0].fill_between(pii[:, 0], mu_z_wc[:, ind_ld_z], mu_z_wc[:, ind_ud_z],
                                  color=colors[0], alpha=.2, lw=3)
    ax[0].axhline(0, color='k', lw=1, linestyle='--')
    ax[0].axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
    ax[0].set_title(r'Symmetric returns', fontsize=15)
    ax[0].set_xlabel(r'$R$', fontsize=15)
    ax[0].legend(loc=2, fontsize=12)


    model = m2

    ind_ld_z, ind_med_z, ind_ud_z = model.ind_ld_z, model.ind_med_z, model.ind_ud_z
    ind_ld_r, ind_med_r, ind_ud_r = model.ind_ld_r, model.ind_med_r, model.ind_ud_r

    pii, rr, zz = model.pii, model.rr, model.zz
    mu_pii_noR, mu_pii, mu_pii_wc = model.mu_pii_noR, model.mu_pii, model.mu_pii_wc
    mu_z, mu_z_wc = model.mu_z, model.mu_z_wc

    inn_r = slice(model.ind_ld_r_noR, model.ind_ud_r_noR)
    ax[1].plot(pii[inn_r, 0], mu_z[inn_r, model.ind_med_z], color='k', lw=3, label="under baseline")
    ax[1].plot(pii[inn_r, 0], mu_z[inn_r, model.ind_ld_z], color='k', lw=2, linestyle='--', alpha=.4)
    ax[1].plot(pii[inn_r, 0], mu_z[inn_r, model.ind_ud_z], color='k', lw=2, linestyle=':', alpha=.4)
    ax[1].fill_between(pii[inn_r, 0], mu_z[inn_r, ind_ld_z], mu_z[inn_r, ind_ud_z],
                                  color='k', alpha=.1, lw=3)

    ax[1].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_med_z], color=colors[0], alpha=.8, lw=3,
                     label="under worst-case")
    ax[1].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], color=colors[0], linestyle='--', lw=2, alpha=.5)
    ax[1].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_ud_z], color=colors[0], linestyle=':', lw=2, alpha=.5)
    ax[1].fill_between(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], mu_z_wc[inn_r, ind_ud_z],
                                  color=colors[0], alpha=.2, lw=3)
    ax[1].axhline(0, color='k', lw=1, linestyle='--')
    ax[1].axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
    ax[1].set_title(r'Asymmetric returns', fontsize=15) #$\mu_Z(R, Z)$', fontsize=15)
    ax[1].set_xlabel(r'$R$', fontsize=15)
    ax[1].set_ylim([-.012, .01])

    plt.tight_layout()

def plot_termstructure(m1, m2, T):
    shock_price_12, shock_price_z = m1.shock_price_12, m1.shock_price_z
    shock_price_12_asym, shock_price_z_asym = m2.shock_price_12, m2.shock_price_z

    fig, ax = plt.subplots(2, 2, figsize=(13, 6.5), sharey=True)

    horizon = np.arange(T)

    ax[0, 0].plot(horizon, shock_price_12[0, :T], color=sns.color_palette()[0], label=r'Symmetric returns')
    ax[0, 0].plot(horizon, shock_price_12[1, :T], color=sns.color_palette()[0], linestyle='--')
    ax[0, 0].plot(horizon, shock_price_12[2, :T], color=sns.color_palette()[0], linestyle=':')
    ax[0, 0].fill_between(horizon, shock_price_12[1, :T], shock_price_12[2, :T], color=sns.color_palette()[0], alpha=.2)
    ax[0, 0].plot(horizon, shock_price_12_asym[0, :T], color=sns.color_palette()[2],
                  label=r'Asymmetric returns')
    ax[0, 0].plot(horizon, shock_price_12_asym[1, :T], color=sns.color_palette()[2], linestyle='--')
    ax[0, 0].plot(horizon, shock_price_12_asym[2, :T], color=sns.color_palette()[2], linestyle=':')
    ax[0, 0].fill_between(horizon, shock_price_12_asym[1, :T], shock_price_12_asym[2, :T],
                          color=sns.color_palette()[2], alpha=.2)
    ax[0, 0].set_ylabel("Shock to capital", fontsize=15)
    ax[0, 0].set_title(r"Varying capital ratio ($Z$ at its median)", fontsize=15)
    ax[0, 0].legend(loc='best', fontsize=12, framealpha=1.)
    ax[0, 0].axhline(0, color='k', lw=1)
    ax[0, 0].set_ylim([0.1, .2])

    ax[0, 1].plot(horizon, shock_price_12[0, :T], color=sns.color_palette()[0], label=r'Symmetric returns')
    ax[0, 1].plot(horizon, shock_price_12[3, :T], color=sns.color_palette()[0], linestyle='--')
    ax[0, 1].plot(horizon, shock_price_12[4, :T], color=sns.color_palette()[0], linestyle=':')
    ax[0, 1].fill_between(horizon, shock_price_12[3, :T], shock_price_12[4, :T], color=sns.color_palette()[0], alpha=.2)
    ax[0, 1].plot(horizon, shock_price_12_asym[0, :T], color=sns.color_palette()[2],
                  label=r'Asymmetric returns')
    ax[0, 1].plot(horizon, shock_price_12_asym[3, :T], color=sns.color_palette()[2], linestyle='--')
    ax[0, 1].plot(horizon, shock_price_12_asym[4, :T], color=sns.color_palette()[2], linestyle=':')
    ax[0, 1].fill_between(horizon, shock_price_12_asym[3, :T], shock_price_12_asym[4, :T],
                          color=sns.color_palette()[2], alpha=.2)
    ax[0, 1].set_title("Varying long run risk state ($R$ at its median)", fontsize=15)
    ax[0, 1].axhline(0, color='k', lw=1)
    ax[0, 1].set_ylim([0.1, .2])

    ax[1, 0].plot(horizon, shock_price_z[0, :T], color=sns.color_palette()[0])
    ax[1, 0].plot(horizon, shock_price_z[1, :T], color=sns.color_palette()[0], linestyle='--')
    ax[1, 0].plot(horizon, shock_price_z[2, :T], color=sns.color_palette()[0], linestyle=':')
    ax[1, 0].fill_between(horizon, shock_price_z[1, :T], shock_price_z[2, :T], color=sns.color_palette()[0], alpha=.2)
    ax[1, 0].plot(horizon, shock_price_z_asym[0, :T], color=sns.color_palette()[2])
    ax[1, 0].plot(horizon, shock_price_z_asym[1, :T], color=sns.color_palette()[2], linestyle='--')
    ax[1, 0].plot(horizon, shock_price_z_asym[2, :T], color=sns.color_palette()[2], linestyle=':')
    ax[1, 0].fill_between(horizon, shock_price_z_asym[1, :T], shock_price_z_asym[2, :T],
                          color=sns.color_palette()[2], alpha=.2)
    ax[1, 0].set_ylabel("Shock to long run risk", fontsize=15)
    ax[1, 0].set_xlabel("Horizon (quarters)", fontsize=15)
    ax[1, 0].axhline(0, color='k', lw=1)
    ax[1, 0].set_ylim([-0.01, .18])

    ax[1, 1].plot(horizon, shock_price_z[0, :T], color=sns.color_palette()[0], label=r'Ex post heterogeneity')
    ax[1, 1].plot(horizon, shock_price_z[3, :T], color=sns.color_palette()[0], linestyle='--')
    ax[1, 1].plot(horizon, shock_price_z[4, :T], color=sns.color_palette()[0], linestyle=':')
    ax[1, 1].fill_between(horizon, shock_price_z[3, :T], shock_price_z[4, :T], color=sns.color_palette()[0], alpha=.2)

    ax[1, 1].plot(horizon, shock_price_z_asym[0, :T], color=sns.color_palette()[2],
                  label=r'Ex ante heterogeneity with $\widehat{\beta}_1=\widehat{\beta}_2$')
    ax[1, 1].plot(horizon, shock_price_z_asym[3, :T], color=sns.color_palette()[2], linestyle='--')
    ax[1, 1].plot(horizon, shock_price_z_asym[4, :T], color=sns.color_palette()[2], linestyle=':')
    ax[1, 1].fill_between(horizon, shock_price_z_asym[3, :T], shock_price_z_asym[4, :T],
                          color=sns.color_palette()[2], alpha=.2)
    ax[1, 1].set_ylim([-0.01, .18])
    ax[1, 1].axhline(0, color='k', lw=1)
    ax[1, 1].set_xlabel("Horizon (quarters)", fontsize=15)
    plt.tight_layout()

def plot_termstructure2(m1, m2, T):
    shock_price_12, shock_price_z = m1.shock_price_12, m1.shock_price_z
    shock_price_12_asym, shock_price_z_asym = m2.shock_price_12, m2.shock_price_z

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    horizon = np.arange(T)

    ax[0].plot(horizon, shock_price_12[0, :T], color=sns.color_palette()[0], label=r'Symmetric returns', lw=2.5)
    #ax[0].plot(horizon, shock_price_12[3, :T], color=sns.color_palette()[0], linestyle='--')
    ax[0].plot(horizon, shock_price_12[4, :T], color=sns.color_palette()[0], linestyle='-.', lw=1)
    ax[0].fill_between(horizon, shock_price_12[3, :T], shock_price_12[4, :T], color=sns.color_palette()[0], alpha=.2)
    ax[0].plot(horizon, shock_price_12_asym[0, :T], color=sns.color_palette()[2],
                  label=r'Asymmetric returns', lw=2.5, linestyle=':')
    #ax[0].plot(horizon, shock_price_12_asym[3, :T], color=sns.color_palette()[2], linestyle='--')
    ax[0].plot(horizon, shock_price_12_asym[4, :T], color=sns.color_palette()[2], linestyle='-.', lw=1)
    ax[0].fill_between(horizon, shock_price_12_asym[3, :T], shock_price_12_asym[4, :T],
                          color=sns.color_palette()[2], alpha=.2)
    #ax[0].set_ylabel("Varying long run risk state ($R$ at its median)", fontsize=15)
    ax[0].set_title(r"Shock to capital", fontsize=15)
    ax[0].legend(loc='best', fontsize=12, framealpha=1.)
    ax[0].axhline(0, color='k', lw=1)
    ax[0].set_ylim([-0.005, .16])
    ax[0].set_xlabel("Horizon (quarters)", fontsize=15)

    ax[1].set_title(r"Shock to long run risk", fontsize=15)
    ax[1].plot(horizon, shock_price_z[0, :T], color=sns.color_palette()[0], label=r'Ex post heterogeneity', lw=2.5)
    #ax[1].plot(horizon, shock_price_z[3, :T], color=sns.color_palette()[0], linestyle='--')
    ax[1].plot(horizon, shock_price_z[4, :T], color=sns.color_palette()[0], linestyle='-.', lw=1)
    ax[1].fill_between(horizon, shock_price_z[3, :T], shock_price_z[4, :T], color=sns.color_palette()[0], alpha=.2)

    ax[1].plot(horizon, shock_price_z_asym[0, :T], color=sns.color_palette()[2],
                  label=r'Ex ante heterogeneity with $\widehat{\beta}_1=\widehat{\beta}_2$', lw=2.5, linestyle=':')
    #ax[1].plot(horizon, shock_price_z_asym[3, :T], color=sns.color_palette()[2], linestyle='--')
    ax[1].plot(horizon, shock_price_z_asym[4, :T], color=sns.color_palette()[2], linestyle='-.', lw=1)
    ax[1].fill_between(horizon, shock_price_z_asym[3, :T], shock_price_z_asym[4, :T],
                          color=sns.color_palette()[2], alpha=.2)
    ax[1].set_ylim([-0.005, .16])
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xlabel("Horizon (quarters)", fontsize=15)

    plt.tight_layout()

def NBER_Shade(ax, start_date):
    """
    This function adds NBER recession bands to a Matplotlib Figure object.
    ax         : axis
    start_date : start date for the sample, form: yyyy-mm-dd
    """

    # load the NBER recession dates
    NBER_Dates = pd.read_csv('./data/NBER_dates.txt')
    sample_1 = pd.Timestamp(start_date) <= pd.DatetimeIndex(NBER_Dates['Peak'])
    sample_2 = pd.Timestamp(start_date) <= pd.DatetimeIndex(NBER_Dates['Trough'])
    NBER_Dates = NBER_Dates[sample_1 + sample_2]

    # for loop generates recession bands!
    for i in NBER_Dates.index:
        ax.axvspan(NBER_Dates['Peak'][i], NBER_Dates['Trough'][i], facecolor='grey', alpha=0.15)

def stationary_uncertaintyprice(x, model):

    N = x.shape[0]
    dens_12 = np.zeros((model.J-2*model.inner, N))
    dens_z = np.zeros((model.J-2*model.inner, N))

    vec_12 = model.h12_vec.reshape(model.J - 2*model.inner, model.I - 2*model.inner)
    density_12 = model.h12_density.reshape(model.J - 2*model.inner, model.I - 2*model.inner)

    vec_z = model.hz_vec.reshape(model.J - 2*model.inner, model.I - 2*model.inner)
    density_z = model.hz_density.reshape(model.J - 2*model.inner, model.I - 2*model.inner)

    for i in range(model.J-2*model.inner):
        f_12 = interp1d(vec_12[i, :], density_12[i, :])
        f_z = interp1d(vec_z[i, :], density_z[i, :])

        in_range_12 = (x > vec_12[i, :].min()) * (x < vec_12[i, :].max())
        in_range_z = (x > vec_z[i, :].min()) * (x < vec_z[i, :].max())

        dens_12[i, in_range_12] = f_12(x[in_range_12])
        dens_z[i, in_range_z] = f_z(x[in_range_z])


    return dens_12.max(0), dens_z.max(0)

def stationary_uncertaintyprice2(density, vec, I):

    length = density.shape[0]
    segment = I - 10
    numb_segment = int(round(length/segment))

    price = np.zeros(numb_segment)
    price_density = np.zeros(numb_segment)

    for i in range(numb_segment):
        price_density[i] = max(density[i*segment:(i+1)*segment])
        dummy = vec[i*segment:(i+1)*segment]
        price[i] = dummy[np.argmax(density[i*segment:(i+1)*segment])]

    return price, price_density

def Figure3():
    if os.path.exists("./data/model_singlecapital.npz"):
        model_1cap = np.load('./data/model_singlecapital.npz')
        sigma_z_1cap = np.load("./data/model_singlecapital_params.npz")['sigma_z']
    else:
        print('Model has not been estimated yet')
        return None
    z_grid = np.linspace(-.8, .6, 100)
    xi_grid = np.zeros(100)
    xi2_grid = np.zeros(100)

    s2 = np.dot(sigma_z_1cap, sigma_z_1cap)
    az_k, ka_k = model_1cap[-5, 6:8]
    az_b, ka_b = model_1cap[-1, 6:8]

    xi0_k, xi1_k, xi2_k = model_1cap[-5, -3:]  # model_asym_HSHS["xi0"],model_asym_HSHS["xi1"]
    xi0_b, xi1_b, xi2_b = model_1cap[-1, -3:]  # model_asym_HSHS2["xi0"],model_asym_HSHS2["xi1"]


    for i, z in enumerate(z_grid):
        xi_grid[i] = xi0_k + 2*xi1_k*z + xi2_k*z**2
        xi2_grid[i] = xi0_b + 2*xi1_b*z + xi2_b*z**2
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z_grid, xi_grid, color='k', lw=2.5, label = r"$\xi^{[\kappa]}(z)$")
    ax.plot(z_grid, xi2_grid, color='gray', lw=2.5, alpha=.8, label = r"$\xi^{[\beta]}(z)$")
    ax.axhline(xi0_k, color='k', linestyle='--', lw=1, alpha=.9)
    ax.axhline(.2**2/2, color='k', linestyle=':', lw=1, alpha=.4)
    ax.text(-.5, .0052, r"$\xi_0$", fontsize=14)
    ax.text(-.5, .017, r"$\frac{\mathsf{q}^2}{2}$", fontsize=15)
    #ax.arrow(-.4, .01, .05, -.002)

    ax.axhline(0, color='k', lw=1, alpha=.8)
    ax.axvline(0, color='k', lw=1, alpha=.8)
    ax.legend(loc=1, fontsize=13, framealpha=1.)
    ax.set_xlabel(r"$Z$", fontsize=14)
    #ax[1].plot(z_grid, norm.pdf(z_grid, loc=az_k/ka_k, scale=np.sqrt(s2/(2*ka_k))))
    #ax[1].plot(z_grid, norm.pdf(z_grid, loc=az_b/ka_b, scale=np.sqrt(s2/(2*ka_b))))
    #ax[1].axhline(0, color='k', lw=1, alpha=.8)
    #ax[1].axvline(0, color='k', lw=1, alpha=.8)

    ax.set_ylim([-.001, .023])
    ax.set_xlim([-.7, .7])
    plt.tight_layout()

def Figure5():
    if os.path.exists("./data/model_sym_HS_p.npz") and os.path.exists("./data/model_asym_HS_p.npz"):
        mm_sym_p = Model(np.load('./data/model_sym_HS_p.npz'))
        mm_asym_p = Model(np.load("./data/model_asym_HS_p.npz"))
    else:
        print('Model has not been estimated yet')
        return None
    model = mm_sym_p
    shock = 0
    dim = 'R'

    if dim=='R':
        dec1 = 1
        dec9 = 2
    elif dim=='Z':
        dec1 = 3
        dec9 = 4

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True) #, sharex=True)

    ax[0].plot(100*model.R_irf[:, 2, shock, 0], lw=3, color='k', label='without robustness')
    #ax[0].plot(100*model.R_irf[:, 2, shock, dec9], lw=1, color='k', linestyle='-.', alpha=.8)
    #ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 2, shock, dec1], 100*model.R_irf[:, 2, shock, 0], 
    #                        color='k', alpha=.1)
    #ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 2, shock, 0], 100*model.R_irf[:, 2, shock, dec9], 
    #                        color='k', alpha=.1)
    ax[0].plot(100*model.R_irf[:, 0, shock, 0], lw=3, color=colors[3], linestyle='--', 
            label='robust control under baseline')
    #ax[0].plot(100*model.R_irf[:, 0, shock, dec9], lw=1, color=colors[3], linestyle='-.', alpha=.8)
    #ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 0, shock, dec1], 100*model.R_irf[:, 0, shock, 0], 
    #                       color=colors[3], alpha=.1)
    #ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 0, shock, 0], 100*model.R_irf[:, 0, shock, dec9], 
    #                       color=colors[3], alpha=.1)
    ax[0].plot(100*model.R_irf[:, 1, shock, 0], lw=3, color=colors[0],  linestyle=':', 
            label='robust control under worst-case')
    #ax[0].plot(100*model.R_irf[:, 1, shock, dec9], lw=1, color=colors[0], linestyle='-.', alpha=.8)
    #ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 1, shock, dec1], 100*model.R_irf[:, 1, shock, 0], 
    #                       color=colors[0], alpha=.1)
    #ax[0].fill_between(np.arange(model.T), 100*model.R_irf[:, 1, shock, 0], 100*model.R_irf[:, 1, shock, dec9], 
    #                       color=colors[0], alpha=.1)
    ax[0].axhline(0, color='k', lw=1)
    ax[0].set_title("Symmetric Returns", fontsize=15)
    ax[0].set_ylabel(r'$R$', fontsize=15, rotation=0)
    ax[0].yaxis.set_label_coords(-0.11, .5)
    #ax[0].set_ylim([-.005, .27])
    ax[0].legend(loc='best', fontsize=12)
    ax[0].set_xlabel("Horizon", fontsize=15)

    model = mm_asym_p

    ax[1].plot(100*model.R_irf[:, 2, shock, 0], lw=3, color='k', label='without robustness')
    #ax[1].plot(100*model.R_irf[:, 2, shock, dec9], lw=1, color='k', linestyle='-.', alpha=.8)
    #ax[1].fill_between(np.arange(model.T), 100*model.R_irf[:, 2, shock, dec1], 100*model.R_irf[:, 2, shock, 0], 
    #                        color='k', alpha=.1)
    #ax[1].fill_between(np.arange(model.T), 100*model.R_irf[:, 2, shock, 0], 100*model.R_irf[:, 2, shock, dec9], 
    #                        color='k', alpha=.1)
    ax[1].plot(100*model.R_irf[:, 0, shock, 0], lw=3, color=colors[3],  linestyle='--', 
            label='robust control under baseline')
    #ax[1].plot(100*model.R_irf[:, 0, shock, dec9], lw=1, color=colors[3], linestyle='-.', alpha=.8)
    #ax[1].fill_between(np.arange(model.T), 100*model.R_irf[:, 0, shock, dec1], 100*model.R_irf[:, 0, shock, 0], 
    #                       color=colors[3], alpha=.1)
    #ax[1].fill_between(np.arange(model.T), 100*model.R_irf[:, 0, shock, 0], 100*model.R_irf[:, 0, shock, dec9], 
    #                       color=colors[3], alpha=.1)
    ax[1].plot(100*model.R_irf[:, 1, shock, 0], lw=3, color=colors[0],  linestyle=':',
            label='robust control under worst-case')
    #ax[1].plot(100*model.R_irf[:, 1, shock, dec9], lw=1, color=colors[0], linestyle='-.', alpha=.8)
    #ax[1].fill_between(np.arange(model.T), 100*model.R_irf[:, 1, shock, dec1], 100*model.R_irf[:, 1, shock, 0], 
    #                       color=colors[0], alpha=.1)
    #ax[1].fill_between(np.arange(model.T), 100*model.R_irf[:, 1, shock, 0], 100*model.R_irf[:, 1, shock, dec9], 
    #                       color=colors[0], alpha=.1)
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_title("Asymmetric Returns", fontsize=15)
    ax[1].set_ylim([-.005, .2]) #.3])
    #ax[1].legend(loc='best')
    ax[1].set_xlabel("Horizon", fontsize=15)

    plt.tight_layout()
    # plt.savefig(figures_path + '2cap_irfR_k2_HS.pdf')

def Figure7():
    if os.path.exists("./data/model_asym_HS_p.npz"):
        mm_asym_p = Model(np.load("./data/model_asym_HS_p.npz"))
    else:
        print('Model has not been estimated yet')
        return None
    model = mm_asym_p

    ind_ld_z, ind_med_z, ind_ud_z = model.ind_ld_z, model.ind_med_z, model.ind_ud_z    
    ind_ld_r, ind_med_r, ind_ud_r = model.ind_ld_r, model.ind_med_r, model.ind_ud_r    

    pii, rr, zz = model.pii, model.rr, model.zz    
    mu_pii_noR, mu_pii, mu_pii_wc = model.mu_pii_noR, model.mu_pii, model.mu_pii_wc
    mu_z, mu_z_wc = model.mu_z, model.mu_z_wc

    #inn_r = slice(model.ind_ld_r_noR, model.ind_ud_r_noR)
    inn_r = slice(1, -1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(pii[inn_r, 0], mu_z[inn_r, model.ind_med_z], color='k', lw=3, label="under baseline")
    #ax[0].plot(pii[inn_r, 0], mu_z[inn_r, model.ind_ld_z], color='k', lw=2, linestyle='--', alpha=.4)
    ax[0].plot(pii[inn_r, 0], mu_z[inn_r, model.ind_ud_z], color='k', lw=1, linestyle='-.', alpha=.4)
    ax[0].fill_between(pii[inn_r, 0], mu_z[inn_r, ind_ld_z], mu_z[inn_r, ind_ud_z], 
                                color='k', alpha=.1, lw=3)
    ax[0].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_med_z], color=colors[0], alpha=.8, lw=3, linestyle=':', 
                    label="under worst-case")
    #ax[0].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], color=colors[0], linestyle='--', lw=2, alpha=.5)
    ax[0].plot(pii[inn_r, 0], mu_z_wc[inn_r, ind_ud_z], color=colors[0], linestyle='-.', lw=1, alpha=.5)
    ax[0].fill_between(pii[inn_r, 0], mu_z_wc[inn_r, ind_ld_z], mu_z_wc[inn_r, ind_ud_z], 
                                color=colors[0], alpha=.2, lw=3)
    ax[0].axhline(0, color='k', lw=1, linestyle='--')
    ax[0].axvline(pii[ind_med_r, 0], color='k', lw=1, linestyle='--')
    ax[0].set_title(r'$\mu_Z(Z, R)$', fontsize=15) 
    ax[0].set_xlabel(r'$R$', fontsize=15)
    ax[0].set_ylim([-.014, .0105])
    ax[0].legend(loc='best', fontsize=12)

    shock=0

    ax[1].plot(model.Z_irf[:600, 1, shock, 0], lw=3, color=colors[0], linestyle=':')
    ax[1].plot(model.Z_irf[:600, 2, shock, 0], lw=3, color='k')
    ax[1].axhline(0, color='k', lw=1, linestyle='--')
    ax[1].set_title("IRF of Z to idiosyncratic capital shock", fontsize=15)
    ax[1].set_xlabel("Horizon", fontsize=15)
    #ax[1].set_ylim([-.001, .01])
    plt.tight_layout()

    # plt.savefig(figures_path + '2cap_asymZ_k2_HS.pdf')

def Figure8():
    if os.path.exists("./data/model_singlecapital.npz"):
        model_1cap = np.load('./data/model_singlecapital.npz')
        states = pd.read_csv('./data/states.csv', index_col=0)

        T = len(states.index)
        z_path = np.asarray(states['filtered_state']).reshape(1, T)

    else:
        print('Single capital stock model has not been estimated yet')
        return None
    #-----------------------------------
    # Unstructured uncertainty prices
    #-----------------------------------
    uncertainty_prices   = np.ones((z_path.shape[1], 1)) @ -model_1cap[3, 17:19].reshape(1, 2)

    #-----------------------------------
    # Structured uncertainty prices
    #-----------------------------------
    H_0_k, H_1_k = model_1cap[-5, 17:19].reshape(2, 1), model_1cap[-5, 19:21].reshape(2, 1)
    H_0_b, H_1_b = model_1cap[-1, 17:19].reshape(2, 1), model_1cap[-1, 19:21].reshape(2, 1)


    uncertainty_prices_k = -(H_0_k + H_1_k @ z_path)
    uncertainty_prices_b = -(H_0_b + H_1_b @ z_path)

    UP = pd.DataFrame(data=np.hstack([uncertainty_prices, 
                                    uncertainty_prices_k.T, 
                                    uncertainty_prices_b.T, 
                                    states.values]), 
                    index=pd.date_range(states.index[0], states.index[-1], freq='Q'), 
                    columns=['price_1_u', 'price_2_u', 'price_1_k', 'price_2_k', 'price_1_b', 'price_2_b', 
                            'cons_growth', 'exp_cons_growth', 'filtered_state'])

    #=================================
    # PLOT
    #=================================
    start_date = '1948-01-31'

    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine()

    # ax.axhline(0, lw=1, color='k', alpha=.9)
    #plt.axhline(uncertainty_prices_unstructured[0], color=colors[7], linestyle='--', lw=2)
    #plt.axhline(uncertainty_prices_unstructured[1], color=colors[7], lw=2)
    UP['price_1_u'].plot(ax=ax, lw=2, linestyle='--', color=colors[7], alpha=.9)
    UP['price_2_u'].plot(ax=ax, lw=2, color=colors[7], alpha=.9)
    UP['price_1_k'].plot(ax=ax, lw=2.5, linestyle='--', color=colors[1], alpha=.9)
    UP['price_2_k'].plot(ax=ax, lw=2.5, color=colors[1], alpha=.9)
    UP['price_1_b'].plot(ax=ax, lw=2.5, color=colors[0], linestyle='--', alpha=.9)
    UP['price_2_b'].plot(ax=ax, lw=2.5, color=colors[0], alpha=.9)


    line1 = mlines.Line2D([], [], color=colors[7], lw=2)
    line2 = mlines.Line2D([], [], color=colors[1], lw=2.5, alpha=.9)
    line3 = mlines.Line2D([], [], color=colors[0], lw=2.5, alpha=.9)

    ax.legend([line1, line2, line3], 
            [r'Constant $\xi$', r'State-dependent $\xi^{[\kappa]}$', r'State-dependent $\xi^{[\beta]}$'], 
            loc='best', fontsize = 12, frameon=True, framealpha=1.0)

    ax.tick_params(axis='both', which='major', labelsize = 10)
    ax.set_ylim([-0.1, .2])

    NBER_Shade(ax, start_date)
    plt.tight_layout()
    # plt.savefig(figures_path + 'uncertainty_prices.pdf')

def Figure9():
    if os.path.exists("./data/model_asym_HSHS2.npz"):
        m1 = Model(np.load("./data/model_sym_HSHS.npz"))
        m2 = Model(np.load("./data/model_asym_HSHS.npz"))
        m1_b = Model(np.load("./data/model_sym_HSHS2.npz"))
        m2_b = Model(np.load("./data/model_asym_HSHS2.npz"))

    # Single capital economy
    stdev_z_1cap = m1.stdev_z_1cap
    x = np.linspace(.05, .35, 1000)
    #h1_cap1 = norm(loc= -H_0[0], scale = H_1[0] * stdev_z_1cap)
    #hz_cap1 = norm(loc= -H_0[1], scale = H_1[1] * stdev_z_1cap)

    # Symmetric capital stocks
    h12_vec, h12_density = stationary_uncertaintyprice2(m1.h12_density, m1.h12_vec, m1.I)
    hz_vec, hz_density = stationary_uncertaintyprice2(m1.hz_density, m1.hz_vec, m1.I)

    h12_b_vec, h12_b_density = stationary_uncertaintyprice2(m1_b.h12_density, m1_b.h12_vec, m1_b.I)
    hz_b_vec, hz_b_density = stationary_uncertaintyprice2(m1_b.hz_density, m1_b.hz_vec, m1_b.I)

    h12_asym_vec, h12_asym_density = stationary_uncertaintyprice2(m2.h12_density, m2.h12_vec, m2.I)
    hz_asym_vec, hz_asym_density = stationary_uncertaintyprice2(m2.hz_density, m2.hz_vec, m2.I)

    h12_asym_b_vec, h12_asym_b_density = stationary_uncertaintyprice2(m2_b.h12_density, m2_b.h12_vec, m2_b.I)
    hz_asym_b_vec, hz_asym_b_density = stationary_uncertaintyprice2(m2_b.hz_density, m2_b.hz_vec, m2_b.I)

    hz_sum = np.sum(h12_density[:-1] * abs(h12_vec[1:]-h12_vec[:-1]))
    hz_asym_sum = np.sum(h12_asym_density[:-1] * abs(h12_asym_vec[1:]-h12_asym_vec[:-1]))

    hz_b_sum = np.sum(h12_b_density[:-1] * abs(h12_b_vec[1:]-h12_b_vec[:-1]))
    hz_asym_b_sum = np.sum(h12_asym_b_density[:-1] * abs(h12_asym_b_vec[1:]-h12_asym_b_vec[:-1]))

    #=============== PLOT =========================#

    fig, ax = plt.subplots(1, 2, figsize = (12, 4), sharex=True, sharey=True)

    ax[0].set_title("Shock to capital", fontsize=16)
    ax[0].plot(h12_vec, h12_density/hz_sum, lw=2, color=colors[1], 
            label=r"Symmetric (concern about $\kappa$)")
    ax[0].fill_between(h12_vec, 0, h12_density/hz_sum, color=colors[1], alpha=.15)
    ax[0].plot(h12_b_vec, h12_b_density/hz_b_sum, lw=2, color=colors[0], 
            label=r"Symmetric returns with $\xi^{[\beta]}$")
    ax[0].plot(h12_asym_vec, h12_asym_density/hz_asym_sum, lw=2, color=colors[1], linestyle=':', 
            label=r'Asymmetric (concern about $\kappa$)')
    ax[0].fill_between(h12_asym_vec, 0, h12_asym_density/hz_asym_sum, color=colors[1], alpha=.1)
    ax[0].plot(h12_asym_b_vec, h12_asym_b_density/hz_asym_b_sum, lw=2, color=colors[0], linestyle=':',
            label=r'Asymmetric (concern about $\beta$)')
    ax[0].fill_between(h12_asym_b_vec, 0, h12_asym_b_density/hz_asym_b_sum, color=colors[0], alpha=.1)
    ax[0].fill_between(h12_b_vec, 0, h12_b_density/hz_b_sum, color=colors[0], alpha=.15)
    ax[0].axhline(0, lw=1.5, color='k')
    ax[0].axvline(0, lw=1, color='k')

    ax[1].set_title("Shock to long run risk state", fontsize=16)
    ax[1].plot(hz_vec, hz_density/hz_sum, lw=2, color=colors[1], 
            label=r"Symmetric with $\xi^{[\kappa]}$")
    ax[1].fill_between(hz_vec, 0, hz_density/hz_sum, color=colors[1], alpha=.15)
    ax[1].plot(hz_asym_vec, hz_asym_density/hz_asym_sum, lw=2, color=colors[1], linestyle=':',
            label=r"Asymmetric with $\xi^{[\kappa]}$")
    ax[1].fill_between(hz_asym_vec, 0, hz_asym_density/hz_asym_sum, color=colors[1], alpha=.1)
    ax[1].plot(hz_b_vec, hz_b_density/hz_b_sum, lw=2, color=colors[0], 
            label=r"Symmetric with $\xi^{[\beta]}$")
    ax[1].plot(hz_asym_b_vec, hz_asym_b_density/hz_asym_b_sum, lw=2, color=colors[0], linestyle=':',
            label=r"Asymmetric with $\xi^{[\beta]}$")
    ax[1].fill_between(hz_asym_b_vec, 0, hz_asym_b_density/hz_asym_b_sum, color=colors[0], alpha=.1)
    ax[1].fill_between(hz_b_vec, 0, hz_b_density/hz_b_sum, color=colors[0], alpha=.15)

    ax[1].axhline(0, lw=1.5, color='k')
    ax[1].legend(loc=2, fontsize=12, ncol=2, frameon=True, framealpha=1.0)
    ax[1].set_xlim([-.12, .17])
    ax[1].axvline(0, lw=1, color='k')

    plt.tight_layout()
    # plt.savefig('local_uncertainty_prices.pdf')


if __name__ == '__main__':
    p = plottingmodule()
    p.intercept_plot()
    p.persistence_plot()
    # p.ex_post_plot()
    # p.ex_ante_animation()
    # p.ex_post_animation()
    # p.twisting_plot()
    # p.sym_r_irf_plot()
    # p.asym_r_irf_plot()
    # p.Z_plot()
    # p.shockplot()

    # code for dumping data
    # plotdata = {}
    # for file_beta in files_beta:
    #     data = {}
    #     file_beta = './data/' + file_beta
    #     model = Model(np.load(file_beta))
    #     f = model.figure_1(left_top_ylim=[.031, .035], numb_lcurves=4)
    #     data['left'] = {}
    #     pii_cut = model.pii[25:-25, 0]
    #     data['left']['x'] = pii_cut
    #     data['left']['withoutr'] = model.d1_noR[25:-25, model.ind_med_z]
    #     data['left']['withoutr_ud'] = model.d1_noR[25:-25, model.ind_ud_z] 
    #     data['left']['withoutr_ld'] = model.d1_noR[25:-25, model.ind_ld_z] 

    #     data['left']['withr'] = model.d1[25:-25, model.ind_med_z]
    #     data['left']['withr_ud'] = model.d1[25:-25, model.ind_ud_z] 
    #     data['left']['withr_ld'] = model.d1[25:-25, model.ind_ld_z] 

    #     data['density'] = {}
    #     data['density']['withoutr'] = f.axes[3].collections[1].get_paths()[0].vertices # 黑中
    #     data['density']['withoutr_lb'] = f.axes[3].collections[2].get_paths()[0].vertices  # 黑内
    #     ite = 0
    #     ll = None
    #     for f_ite in f.axes[3].collections[0].get_paths():
    #         if ll == None:
    #             candidate = f_ite.vertices
    #             ll = len(f_ite)
    #         else:
    #             if len(f_ite) > ll:
    #                 ll = len(f_ite)
    #                 candidate = f_ite.vertices

    #     data['density']['withoutr_ub'] = candidate  # 黑外
    #     data['density']['r_base'] = f.axes[3].collections[6].get_paths()[0].vertices  # 红中
    #     data['density']['r_base_lb'] = f.axes[3].collections[7].get_paths()[0].vertices # 红内
    #     data['density']['r_base_ub'] = f.axes[3].collections[5].get_paths()[0].vertices # 红外
    #     data['density']['r_worst'] = f.axes[3].collections[11].get_paths()[0].vertices # 蓝中
    #     data['density']['r_worst_lb'] = f.axes[3].collections[12].get_paths()[0].vertices # 蓝内
    #     data['density']['r_worst_ub'] = f.axes[3].collections[10].get_paths()[0].vertices # 蓝外

    #     data['top'] = {}
    #     data['top']['x'] = model.pii[1:,0]
    #     data['top']['f_without'] = model.f_noR_dist.sum(1)
    #     data['top']['f_base'] = model.f_dist.sum(1)
    #     data['top']['f_worst'] = model.f_wc_dist.sum(1)

    #     data['right'] = {}
    #     data['right']['x'] = model.zz[0,:]
    #     data['right']['f_without'] = model.f_noR_dist.sum(0)
    #     data['right']['f_base'] = model.f_dist.sum(0)
    #     data['right']['f_worst'] = model.f_wc_dist.sum(0)
        
    #     if file_beta[-6] == '0':
    #         plotdata[np.round(np.int(file_beta[-6:-4]) * 1e-2, 3)] = data
    #     else:
    #         plotdata[np.round(np.int(file_beta[-6:-4]) * 1e-3, 3)] = data