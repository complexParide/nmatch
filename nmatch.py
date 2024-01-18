####################################################################################
# 
# MIT License
# 
# Copyright (c) 2024 Paride Crisafulli
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
####################################################################################


# imports and matplotlib settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import pickle
from tqdm import tqdm
from multiprocessing import cpu_count
import time
import pymc3 as pm
import arviz as az
plt.style.use('bmh')


# THE MAIN CLASS OF THE MODULE
class NMAtchModel():
    
    # initializes the NMAtch model
    def __init__(self, edgelist, edgelist_map=None):
        # edgelist: pd.DataFrame containing four columns: HomeTeam, AwayTeam, HomeScore, AwayScore
        # edgelist_map: maps the name of these four columns in case they are different
        if edgelist_map == None:
            self.edgelist_map = {'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam',
                                 'HomeScore': 'HomeScore', 'AwayScore': 'AwayScore'}
        elif any(('HomeTeam' not in edgelist_map.keys(), 'HomeScore' not in edgelist_map.keys(),
                  'AwayTeam' not in edgelist_map.keys(),'AwayScore' not in edgelist_map.keys())):
            raise Exception('Ill-defined edgelist_map!')
        else:
            self.edgelist_map = edgelist_map
        self.edgelist = edgelist
        self.treatments = np.unique(np.concatenate((edgelist[self.edgelist_map['HomeTeam']],
                                                    edgelist[self.edgelist_map['AwayTeam']])))
        self.n_d = self.treatments.shape[0]
        self.d_to_idx = dict(zip(self.treatments, range(self.n_d)))
        self.idx_to_d = dict(zip(range(self.n_d), self.treatments))
        self.n_trials = self.edgelist.shape[0]
        self.model = None
        
    # builds up the NMAtch model
    def modelBuilder(self, home_boost=True):
        # home_boost: boolean that sets beta automatically to 1 if False
        self.model = pm.Model()
        with self.model:
            # ATT and DEF variables
            atts = pm.Uniform("attacks", lower=0, upper=1, shape=self.n_d)
            defs = pm.Uniform("defenses", lower=1e-10, upper=1, shape=self.n_d)
            if home_boost:
                # beta variable
                hb = pm.TruncatedNormal("home_boost", mu=1, sigma=5, lower=0)
            for mi, line in tqdm(self.edgelist.iterrows(), colour='green',
                                 total=self.edgelist.shape[0]):
                hidx = self.d_to_idx[line[self.edgelist_map['HomeTeam']]]
                aidx = self.d_to_idx[line[self.edgelist_map['AwayTeam']]]
                if home_boost:
                    homescore = pm.Poisson(f"homescore_{mi}", mu=hb * atts[hidx]/defs[aidx],
                                           observed=line[self.edgelist_map['HomeScore']])
                else:
                    homescore = pm.Poisson(f"homescore_{mi}", mu=atts[hidx]/defs[aidx],
                                           observed=line[self.edgelist_map['HomeScore']])
                awayscore = pm.Poisson(f"awayscore_{mi}", mu=atts[aidx]/defs[hidx],
                                       observed=line[self.edgelist_map['AwayScore']])
    
    # actually runs the NMAtch model
    def runBayes(self, niters=2500, burn_in=1000, chains=4, save_path=None, home_boost=False):
        # niters: number of MCMC iterations
        # burn_in: number of discarded MCMC initial iterations
        # chains: number of MCMC chains
        # save_path: if not None saves the PyMC3 trace
        # home_boost: boolean that sets beta automatically to 1 if False
        startime = time.time()
        self.modelBuilder(home_boost=home_boost)
        print('Starting to sample (this will take a while, especially after the prog-bar)...')
        with self.model:
            trace = pm.sample(niters, tune=burn_in, cores=min((cpu_count(), chains)),
                              chains=chains, return_inferencedata=True)
        print('Done!')
        elapstime = (time.time() - startime) // 60
        print(f"\nBayesian sampling is over! [tte = {elapstime} mins]")
        if save_path is not None:
            pickle.dump(trace, open(save_path, 'wb'))
        return trace
    
# creates a pd.DataFrame with the results of the inference (can save a csv)
def resultTable(trace, d_names, save_path=None, show=True, home_boost=False):
    # trace: output of the runBayes function
    # d_names: names of the teams
    # save_path: if not None saves the table
    # show: if True displays the table
    # home_boost: boolean that sets beta automatically to 1 if False
    if home_boost:
        df = az.summary(trace)
        namecol = pd.DataFrame(np.column_stack((np.concatenate((np.repeat("att", len(d_names)),
                                                                np.repeat("def", len(d_names)),
                                                                ['home_boost'])),
                                                np.concatenate((d_names, d_names,
                                                                ['home_boost'])))),
                               columns=['variable', 'team'])
    else:
        df = az.summary(trace, var_names=['attacks', 'defenses'])
        namecol = pd.DataFrame(np.column_stack((np.concatenate((np.repeat("att", len(d_names)),
                                                                np.repeat("def", len(d_names)))),
                                                np.concatenate((d_names, d_names)))),
                               columns=['variable', 'team'])
    df = pd.concat((namecol, df.reset_index(drop=True)), axis=1)
    if save_path is not None:
        df.to_csv(save_path, index=None)
    if show:
        display(df)
    return df

# creates forest plots for ATT and DEF
def forestPlot(pymc_table, att_color='xkcd:crimson', def_color='xkcd:sapphire', show=True,
               att_save_path=None, def_save_path=None, xlim=None, sucras=None, fontsize=17,
               att_title='', def_title=''):
    # pymc_table: output of resulTable
    # att_color: color of the ATT bars in the forest plot
    # def_color: color of the DEF bars in the forest plot
    # show: if True displays the forest plots
    # att_save_path: if not None saves the ATT forest plot
    # def_save_path: if not None saves the DEF forest plot
    # xlim: limit of the x axis for the DEF forest plot (the ATT one is 1)
    # sucras: output of rankoTeam to sort the teams according to SUCRA.
    #         If None they are sorted alphabetically
    # fontsize: fontsize of the axes ticklabels
    # att_title: title of the ATT forest plot
    # def_title: title of the DEF forest plot
    colodict = {'attack': att_color, 'defense': def_color}
    savedict = {'attack': att_save_path, 'defense': def_save_path}
    titldict = {'attack': att_title, 'defense': def_title}
    for var in ['attack', 'defense']:
        df = pymc_table.copy()[pymc_table['variable'] == var[:3]]
        figa, ax = plt.subplots(1, 1, figsize=(12, df.shape[0]))
        if sucras is not None:
            df['sucras'] = sucras['sucra'].values
            df = df.sort_values(by='sucras')
        hdi = 0.5 * (df['hdi_97%'] - df['hdi_3%'])
        ax.errorbar(df['mean'], range(df.shape[0]), xerr=df['sd'], zorder=10,
                    color=colodict[var], linewidth=5, linestyle='')
        ax.errorbar(df['mean'], range(df.shape[0]), xerr=hdi, color=colodict[var],
                    linewidth=2.5, zorder=5, linestyle='')
        ax.scatter(df['mean'], range(df.shape[0]), marker='o', s=150, facecolor='white',
                   edgecolor=colodict[var], zorder=50, linewidth=1.5)
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels(df['team'], fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.set_title(titldict[var], fontsize=fontsize)
        if var == 'attack':
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(0, xlim)
        if savedict[var] is not None:
            plt.savefig(savedict[var], bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(figa)

            
# generates rankograms and SUCRA tables
def rankoTeam(edgelist, trace, teams=None, cmap='plasma_r', fontsize=12, edgelist_map=None,
              starting_points=None, save_path=None, ranko_show=False, sucra_show=True,
              progbar_desc=''):
    # edgelist: pd.DataFrame of the matches (training ones should be excluded)
    #           Analogous to NMAtchModel __init__
    # trace: runBayes output
    # teams: names of the teams in the tournament
    # cmap: colormap of the rankograms
    # fontsize: fontsize of the axes ticklabels
    # edgelist_map: maps the name of these four columns in case they are different
    # starting_points: starting points of each team in the league. If None they start from zero
    # save_path: if not None saves the rankograms and the SUCRA table
    # ranko_show: if True displays the rankograms
    # sucra_show: if True displays the SUCRA table
    # progbar_desc: description of the progbar (tqdm)
    if edgelist_map == None:
        edgelist_map = {'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam',
                        'HomeScore': 'HomeScore', 'AwayScore': 'AwayScore'}
    if teams is None:
        teams = np.unique(np.concatenate((edgelist[edgelist_map['HomeTeam']].values,
                                          edgelist[edgelist_map['AwayTeam']].values)))
    if starting_points is None:
        starting_points = dict(zip(teams, np.zeros(len(teams))))
    if len(teams) != len(starting_points):
        raise Exception('Teams list of different lenght than starting_points!')
    attack_samples = dict(zip(teams, np.concatenate(trace.posterior['attacks'].values,
                                                    axis=0).T))
    defense_samples = dict(zip(teams, np.concatenate(trace.posterior['defenses'].values,
                                                     axis=0).T))
    home_boosts =  np.concatenate(trace.posterior['home_boost'].values)
    shape = np.sum(home_boosts.shape[0])
    ranks = np.zeros((shape, len(teams)))
    for i in tqdm(range(shape), colour='yellow', desc=progbar_desc, ncols=100):
        points = starting_points.copy()
        for _, line in edgelist.iterrows():
            hpred = home_boosts[i] * attack_samples[line[edgelist_map['HomeTeam']]][i] /\
                    defense_samples[line[edgelist_map['AwayTeam']]][i]
            apred = attack_samples[line[edgelist_map['AwayTeam']]][i] /\
                    defense_samples[line[edgelist_map['HomeTeam']]][i]
            hpred = np.random.poisson(hpred)
            apred = np.random.poisson(apred)
            if hpred > apred:
                points[line[edgelist_map['HomeTeam']]] += 3
            elif hpred < apred:
                points[line[edgelist_map['AwayTeam']]] += 3
            else:
                points[line[edgelist_map['HomeTeam']]] += 1
                points[line[edgelist_map['AwayTeam']]] += 1
        sorted_teams = sorted(teams, key=lambda x: -points[x])
        ranks[i] = [sorted_teams.index(team) for team in teams]
    sucras = np.zeros(ranks.shape[1])
    rankomat = np.zeros((len(teams), len(teams)))
    for i in range(ranks.shape[1]):
        vals, counts = np.unique(ranks[:, i], return_counts=True)
        vals = vals + 1
        figa, ax = plt.subplots(1, 1, figsize=(6, 4))
        col = np.average(vals, weights=counts) / max(vals)
        probs = counts / np.sum(counts)
        sucra = np.cumsum(probs)[:-1]
        sucras[i] = np.sum(sucra) / len(sucra)
        for v, p in zip(vals, probs):
            rankomat[int(v-1)] = p
        ax.bar(vals, probs, color=colormaps[cmap](col), zorder=10, label=teams[i].capitalize())
        ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
        ax.set_yticks(np.arange(0, 1.01, 0.2), minor=False)
        ax.set_yticklabels(np.round(np.arange(0, 1.01, 0.2), 1), fontsize=fontsize)
        ax.set_xticks(np.arange(2, ranks.shape[1] + 1, 2), minor=False)
        ax.set_xticks(np.arange(1, ranks.shape[1] + 1), minor=True)
        ax.set_xticklabels(np.arange(2, ranks.shape[1] + 1, 2), fontsize=fontsize)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, ranks.shape[1] + 1)
        legend = ax.legend(loc='best', fontsize=fontsize*1.25)
        for text in legend.get_texts():
            text.set_horizontalalignment('center')
        if save_path is not None:
            plt.savefig(f"{save_path}/rkg__{teams[i].replace(' ', '_')}.pdf", bbox_inches='tight')
        if ranko_show:
            plt.show()
        else:
            plt.close(figa)
    sucra_df = pd.DataFrame(np.column_stack((teams, sucras)), columns=['treatment', 'sucra'])
    if sucra_show:
        display(sucra_df)
    if save_path is not None:
        sucra_df.to_csv(f"{save_path}sucra_scores.csv", index=None)
    return rankomat, sucra_df
