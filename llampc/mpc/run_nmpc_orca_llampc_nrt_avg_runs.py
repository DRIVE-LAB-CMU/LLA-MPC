"""	Nonlinear MPC using dynamic bicycle model with real-time model adaptation.
"""
# [([8.32, 8.18, 8.379999999999999, 9.059999999999999, 0.0], 0.0, 0.04420309176883905),
# ([7.82, 7.68, 7.780000000000001, 8.379999999999999, 0.0], 0.0, 0.07711016808712949),
# ([8.32, 8.120000000000001, 8.82, 9.419999999999998, 0.0], 0.0, 0.05007025317138445),
# ([8.120000000000001, 7.84, 8.100000000000001, 8.659999999999997, 0.0], 0.0, 0.05069589589237952),
# ([8.08, 7.880000000000001, 8.04, 8.619999999999997, 0.0], 0.0, 0.04642332823640459),
# ([8.48, 8.260000000000002, 8.939999999999998, 9.5, 0.0], 0.4, 0.033198900728473014),
# ([8.44, 8.22, 8.280000000000001, 8.559999999999999, 0.0], 2.78, 0.07206404684217566),
# ([8.06, 7.720000000000001, 7.919999999999998, 8.379999999999999, 0.0], 3.96, 0.09176545652261256),
# ([8.76, 8.660000000000002, 8.82, 0.0, 0.0], 0.38, 0.03081686945432081),
# ([7.84, 7.66, 8.100000000000001, 8.759999999999998, 0.0], 1.3199999999999998, 0.05631315401538816)]


# ]

def calculate_mean_of_indices(tuples_list):
    # Extract the first elements (the lists) and the rest of the elements (the individual numbers)
    lists = [t[0] for t in tuples_list]  # The lists inside each tuple
    numbers = [t[1:] for t in tuples_list]  # The individual numbers in each tuple

    # Convert lists to numpy arrays and calculate the mean of corresponding indices
    lists_mean = np.mean(lists, axis=0)  # Mean of the lists
    numbers_mean = np.mean(numbers, axis=0)  # Mean of the individual numbers

    # Combine the means into a final result
    return np.concatenate((lists_mean, numbers_mean))

__author__ = 'Maitham AL-Sunni'
__email__ = 'maitham@cmu.edu'
from llampc.mpc.constraints import Boundary

import time as tm
import numpy as np
import casadi
import _pickle as pickle

import matplotlib
from matplotlib.lines import lineStyles

matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# matplotlib.use("pgf")  # Uses a LaTeX-compatible backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import subprocess


from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

from llampc.params import ORCA
from llampc.models import Dynamic
from llampc.tracks import ETHZ, ETHZMobil
from llampc.mpc.planner import ConstantSpeed
from llampc.mpc.nmpc import setupNLP
import multiprocessing as mp
from llampc.mpc.evaluate_models_vectorized import evaluate_models_vectorized
import os
import imageio
import copy

import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True

# n_cores = mp.cpu_count()
# print("number of cores is: ",n_cores)
# pool = mp.Pool(processes=1)

# joblib
# pool = Parallel(n_jobs=n_cores)
# print("number of cores: ",n_cores)

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
TRACK_CONS = False

#####################################################################
# default settings
SIM_TIME = 28
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

if not TRACK_CONS:
    SUFFIX = 'NOCONS-'
else:
    SUFFIX = ''

ALL_TRYS = []
for TRYING in range(1):
    violation_total = []
#####################################################################
# load vehicle parameters

    params = ORCA(control='pwm')
    model = Dynamic(**params)
    print("check1",id(model))

    model_run = Dynamic(**params)

    #####################################################################
    # Model Bank Setup

    N_MODELS = 10000  # number of models in the bank
    N_AC_STEPS = 10  # Number of steps to accumulate error
    smoothing_mu = 20 # moving avg for MUs
    smoothing_mu_over_mod = 10 # getting the avg MUs for the best N models
    v_factor = .9
    mu_init = 1.

    class ExponentialSmoother:
        def __init__(self, alpha=0.3):  # Alpha controls smoothness (0.1-0.3 recommended)
            self.alpha = alpha
            self.smooth_value = None

        def update(self, new_value):
            if self.smooth_value is None:
                self.smooth_value = new_value  # Initialize with first value
            else:
                self.smooth_value = self.alpha * new_value + (1 - self.alpha) * self.smooth_value
            return self.smooth_value  # Return smoothed value

    # tune it!!!!!!!!!!!
    # 0.2 for sudden is good (but osc.). 0.03 is smooth but slower. 0.06 is faster a lil bit.

    smoother = ExponentialSmoother(alpha=0.08)


    def find_closest_point(x, y, raceline):
        # Get raceline coordinates
        x_refs = raceline[0]
        y_refs = raceline[1]
        # Calculate distances to all points
        distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)
        # Find index of minimum distance
        idx = np.argmin(distances)
        return idx, x_refs[idx], y_refs[idx], distances[idx]

    def update_friction(Df,Dr,curr_time,style='const_decay') :
        if style is 'const_decay' :
            if curr_time > 5:
                Df -= Df/2600.
                Dr -= Dr/2600.
        elif style is 'sudden' :
            if curr_time > 5.3 and curr_time < 5.5:
            # if curr_time > 10.3 and curr_time < 10.5:
                Df -= Df/22.
                Dr -= Dr/22.
        elif style is 'no_change' :
            return Df, Dr
        return Df, Dr

    MODEL_BANK = []
    MODEL_PARAMS = []
    laps_completed = 0
    lap_times = [0.,0.,0.,0.,0.]

    Drs = []
    Dfs = []

    Drs_preds = []
    Dfs_preds = []

    MUs = []
    MU_preds = []


    # Define variation percentages for each parameter
    variation_dict = {
        # 'mass': 0.15,        # 20% variation
        # 'lf': 0.15,          # 10% variation
        # 'lr': 0.15,          # 10% variation
        # 'Cm1': 0.15,         # 15% variation
        # 'Cm2': 0.15,         # 15% variation
        # 'Cr0': 0.15,         # 15% variation
        # 'Cr2': 0.15,         # 15% variation
        'Br': 1.5,          # 15% variation
        'Cr': 1.5,         # 15% variation
        'Dr': 1.5,          # 15% variation
        'Bf': 1.5,          # 15% variation
        'Cf': 1.5,          # 15% variation
        'Df': 1.5,          # 15% variation
    }

    fval_history = []


    # Create model bank with parameter variations
    for i in range(N_MODELS):
        param_variation = params.copy()

        for param_name, variation_percentage in variation_dict.items():
            if param_name in param_variation:
                param_variation[param_name] *= (1 + variation_percentage * np.random.randn())

        MODEL_PARAMS.append(param_variation)
        MODEL_BANK.append(Dynamic(**param_variation))

    Bfs_pass = np.array([m.Bf for m in MODEL_BANK])
    Cfs_pass = np.array([m.Cf for m in MODEL_BANK])
    Dfs_pass = np.array([m.Df for m in MODEL_BANK])
    Brs_pass = np.array([m.Br for m in MODEL_BANK])
    Crs_pass = np.array([m.Cr for m in MODEL_BANK])
    Drs_pass = np.array([m.Dr for m in MODEL_BANK])

    params_pass = (Bfs_pass, Cfs_pass, Dfs_pass, Brs_pass, Crs_pass, Drs_pass)

    #####################################################################
    # load track

    TRACK_NAME = 'ETHZ'
    track = ETHZMobil(reference='optimal', longer=True)

    #####################################################################
    # extract data

    Ts = SAMPLING_TIME
    n_steps = int(SIM_TIME/Ts)
    n_states = model.n_states
    n_inputs = model.n_inputs
    horizon = HORIZON

    # Initialize error windows as numpy array
    error_windows = np.zeros((N_MODELS, N_AC_STEPS))
    window_count = 0

    # Setup NLP solvers for all models
    # print("Setting up NLP solvers for all models...")
    # nlp_bank = []
    # for i in range(N_MODELS):
    #     print("setting up NLP # {}".format(i))
    #     nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R,
    #                   MODEL_PARAMS[i],
    #                   MODEL_BANK[i],
    #                   track, track_cons=TRACK_CONS)
    #     nlp_bank.append(nlp)
    # print("NLP setup complete.")

    # just begainning

    nlp_initial = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_run, track, track_cons=TRACK_CONS)

    # nlp_initial = nlp_bank[0]

    #####################################################################
    # closed-loop simulation

    # initialize
    ref_speeds = []

    states = np.zeros([n_states, n_steps+1])
    dstates = np.zeros([n_states, n_steps+1])
    inputs = np.zeros([n_inputs, n_steps])
    time = np.linspace(0, n_steps, n_steps+1)*Ts

    Ffy = np.zeros([n_steps+1])
    Frx = np.zeros([n_steps+1])
    Fry = np.zeros([n_steps+1])

    Ffy_preds = np.zeros([n_steps+1])
    Frx_preds = np.zeros([n_steps+1])
    Fry_preds = np.zeros([n_steps+1])

    hstates = np.zeros([n_states,horizon+1])
    hstates2 = np.zeros([n_states,horizon+1])

    Hs0 = []
    Hs1 = []

    Hs0_2 = []
    Hs1_2 = []

    # Arrays to store model performance metrics
    model_switches = []  # Store timesteps when model switches occur
    model_mses = []     # Store MSE values at switches
    chosen_models = []  # Store which model was chosen

    projidx = 0
    x_init = np.zeros(n_states)
    x_init[0], x_init[1] = track.x_init, track.y_init
    x_init[2] = track.psi_init
    x_init[3] = track.vx_init
    dstates[0,0] = x_init[3]
    states[:,0] = x_init
    print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))


    media_dir = "LLA"
    os.makedirs(media_dir, exist_ok=True)

    # dynamic plot
    H = .1
    W = .05
    dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])

    fig_track = track.plot(color='k', grid=False)
    fig_track.set_dpi(50)
    plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
    ax = plt.gca()
    LnS, = ax.plot(states[0,0], states[1,0], '#4B0082', label='Trajectory',alpha=0.65)
    # LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=.5, lw=0.5, label="reference")
    xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
    LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
            , states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8, label='Current pose')
    LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
    LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")
    plt.xlabel(r'$x$ [$\mathrm{m}$]')
    plt.ylabel(r'$y$ [$\mathrm{m}$]')
    plt.legend()

    # Initialize model selection variables
    current_model_idx = 0  # Start with nominal model
    uprev = np.zeros(n_inputs)  # Initialize uprev


    # main simulation loop
    for idt in range(n_steps-horizon):
        print("checkiter", id(model))
        x0 = states[:,idt]

        #  "const_decay" or "sudden" or "no_change"
        model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts, "sudden")
        params['Df'], params['Dr'] = model.Df, model.Dr

        # planner based on BayesOpt
        if idt > N_AC_STEPS+1:
            xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx,
                                             curr_mu=MU_pred, scale=v_factor)
        else:
            xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

        ref_speeds.append(v)
        fval_history.append(find_closest_point(x0[0], x0[1], track.raceline)[-1])

        # if projidx > 656:
        if projidx > 440:
            if laps_completed > 0:
                lap_times[laps_completed] = idt * Ts
                print(lap_times)
            else:
                lap_times[laps_completed] = idt * Ts
                print(lap_times)
            laps_completed += 1
            projidx = 0

        start = tm.time()
        # Use pre-computed NLP solver
        # if idt <= N_AC_STEPS:
        #     chosen_model = model
        # else:
        #     print("\n=== Previous Model Parameters ===")
        #     print(f"Model ID: {current_model_idx}")
        #     print(f"Bf: {chosen_model.Bf}")
        #     print(f"Cf: {chosen_model.Cf}")
        #     print(f"Df: {chosen_model.Df}")
        #     print(f"Br: {chosen_model.Br}")
        #     print(f"Cr: {chosen_model.Cr}")
        #     print(f"Dr: {chosen_model.Dr}")
        #     chosen_model = MODEL_BANK[current_model_idx]
        #     print("\n=== New Model Parameters ===")
        #     print(f"Model ID: {current_model_idx}")
        #     print(f"Bf: {chosen_model.Bf}")
        #     print(f"Cf: {chosen_model.Cf}")
        #     print(f"Df: {chosen_model.Df}")
        #     print(f"Br: {chosen_model.Br}")
        #     print(f"Cr: {chosen_model.Cr}")
        #     print(f"Dr: {chosen_model.Dr}")
        #     # nlp = nlp_bank[current_model_idx]

        if idt <= N_AC_STEPS:
            nlp = nlp_initial
        else:
            nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R,
                      MODEL_PARAMS[current_model_idx],
                      MODEL_BANK[current_model_idx],
                      track, track_cons=TRACK_CONS)

        umpc, fval, xmpc, violation = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)

        inputs[:,idt] = umpc[:,0]
        uprev = inputs[:,idt]  # Update uprev for next iteration
        # print("iter: {}, model: {}, cost: {:.5f}, time: {:.2f}, mse: {:.5f}".format(
        #     idt,
        #     current_model_idx,
        #     fval,
        #     end - start,
        #     np.mean(error_windows[current_model_idx])*10000 if window_count > 0 else 0.0
        # ))

        # update current position with numerical integration (exact model)
        x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
        Ain, bin = Boundary(np.array(x_next[:2, -1]), track)
        flag = (np.array(Ain @ np.array(x_next[:2, -1])).T > np.array(np.array([bin[0][0], bin[1][0]]))).any()
        violation_total.append(flag * 0.02)

        states[:,idt+1] = x_next[:,-1]
        dstates[:,idt+1] = dxdt_next[:,-1]
        Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])
        Ffy_preds[idt+1], Frx_preds[idt+1], Fry_preds[idt+1] = MODEL_BANK[current_model_idx].calc_forces(states[:,idt], inputs[:,idt], return_slip=False)

        Drs.append(model.Dr)
        Dfs.append(model.Df)

        # Drs_preds.append(MODEL_BANK[current_model_idx].Dr)
        # Dfs_preds.append(MODEL_BANK[current_model_idx].Df)

        if idt <= N_AC_STEPS:
            Drs_preds.append(mu_init * params['mass'] * 9.8 * params['lr'] / (params['lf'] + params['lr']))
            Dfs_preds.append(mu_init * params['mass'] * 9.8 * params['lf'] / (params['lf'] + params['lr']))
            MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))
            MU_preds.append(copy.deepcopy(mu_init))
        else:
            MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))

            bestKDr = []
            bestKDf = []
            for best_ind in ind_best_KM:
                bestKDr.append(copy.deepcopy(MODEL_BANK[best_ind].Dr))
                bestKDf.append(copy.deepcopy(MODEL_BANK[best_ind].Df))
            Drs_preds.append(np.mean(bestKDr))
            Dfs_preds.append(np.mean(bestKDf))
            MU_pred = copy.deepcopy(np.mean(np.array(Drs_preds)[-smoothing_mu:])  + np.mean(np.array(Dfs_preds)[-smoothing_mu:]) ) / (9.81 * params['mass'])

            # .95 workes perfectly.
            MU_preds.append(smoother.update(MU_pred)*.95)

        # In your main loop:
        if idt > 0:
            # Calculate errors for all models at once

            errors = np.mean((evaluate_models_vectorized(MODEL_BANK, N_MODELS, states[:, idt], inputs[:, idt], Ts, params_pass) - states[0:4, idt + 1]) ** 2, axis=1)

            # Update error windows using numpy operations
            error_windows = np.roll(error_windows, -1, axis=1)
            error_windows[:, -1] = errors
            window_count = min(window_count + 1, N_AC_STEPS)

            # Make decision if we have enough data points
            if window_count >= N_AC_STEPS:
                avg_errors = np.mean(error_windows, axis=1)
                new_model_idx = np.argmin(avg_errors)
                ind_best_KM = avg_errors.argsort()[:smoothing_mu_over_mod]

                if new_model_idx != current_model_idx:
                    current_model_idx = new_model_idx
                    model_switches.append(idt)
                    model_mses.append(avg_errors[new_model_idx])
                    chosen_models.append(current_model_idx)

        # forward sim to predict over the horizon
        hstates[:,0] = x0
        hstates2[:,0] = x0
        for idh in range(horizon):
            x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
            hstates[:,idh+1] = x_next[:,-1]
            hstates2[:,idh+1] = xmpc[:,idh+1]

        Hs0.append(copy.deepcopy(hstates[0]))
        Hs1.append(copy.deepcopy(hstates[1]))

        Hs0_2.append(copy.deepcopy(hstates2[0]))
        Hs1_2.append(copy.deepcopy(hstates2[1]))

        end = tm.time()
        print("iter: {}, model: {}, cost: {:.5f}, time: {:.2f}, mse*10000: {:.5f}".format(
            idt,
            current_model_idx,
            fval,
            end - start,
            np.mean(error_windows[current_model_idx])*10000 if window_count > 0 else 0.0
        ))

        mean_cost = np.mean(fval_history)
        print(f"Mean cost at iter {idt}: {mean_cost:.3f}")



    for i in range(len(lap_times)-1,0,-1) :
        if lap_times[i] != 0. :
            lap_times[i] = lap_times[i] - lap_times[i-1]
    print(lap_times)
    print("lap times:",lap_times, "violation: ",np.sum(violation_total), "mean_dev", np.mean(fval_history))
    ALL_TRYS.append((lap_times, np.sum(violation_total), np.mean(fval_history)))
    print(ALL_TRYS)

ALL_TRYS = calculate_mean_of_indices(ALL_TRYS)
print(ALL_TRYS)

