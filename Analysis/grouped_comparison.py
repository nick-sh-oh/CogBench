"""
Produces grouped vertical bar charts and an MDS embedding plot for comparing
models on CogBench, in the style of the Centaur paper (Binz et al., Nature 2025).

Usage:
    cd Analysis
    python3 grouped_comparison.py --models socius/Llama-Centaur-1B unsloth/Llama-3.2-1B \
        [--interest behaviour|performance|both] [--embedding] [--store_id my_run]

Outputs (saved to ./plots/grouped/):
    - {store_id}_performance.pdf   (plot b: grouped bars for performance)
    - {store_id}_behaviour.pdf     (plot c: grouped bars for behavioural metrics)
    - {store_id}_embedding.pdf     (plot a: MDS 2D embedding with labels)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from utils import merge_all_metrics_and_features


# ── shared helpers ──────────────────────────────────────────────────────────

def _exclude_agents():
    excluding = ['rational', 'meta-RL']
    excluding += [
        'llama_65', 'llama_30', 'llama_13', 'llama_7',
        'vicuna_13', 'vicuna_7',
        'hf_vicuna-7b-v1.3', 'hf_vicuna-13b-v1.3', 'hf_vicuna-33b-v1.3',
        'hf_koala-7B-HF', 'hf_koala-13B-HF',
    ]
    df_llms = pd.read_csv('./data/llm_features.csv')
    for engine in df_llms['Engine'].unique().tolist():
        if engine.endswith('_cot') or engine.endswith('_sb'):
            excluding.append(engine)
    return excluding


BEHAVIOUR_EXPERIMENTS = {
    'ProbabilisticReasoning': ['behaviour_score1', 'behaviour_score2'],
    'HorizonTask': ['behaviour_score1', 'behaviour_score2'],
    'RestlessBandit': ['behaviour_score3'],
    'InstrumentalLearning': ['behaviour_score1', 'behaviour_score2'],
    'TwoStepTask': ['behaviour_score1'],
    'TemporalDiscounting': ['performance_score1'],
    'BART': ['behaviour_score1'],
}
BEHAVIOUR_NAMES = [
    'Prior weighting', 'Likelihood weighting',
    'Directed exploration', 'Random exploration',
    'Meta-cognition', 'Learning rate', 'Optimism bias',
    'Model-basedness', 'Temporal discounting', 'Risk taking',
]

PERFORMANCE_EXPERIMENTS = {
    'ProbabilisticReasoning': ['performance_score1'],
    'HorizonTask': ['performance_score1'],
    'RestlessBandit': ['performance_score1'],
    'InstrumentalLearning': ['performance_score1'],
    'TwoStepTask': ['performance_score1'],
    'BART': ['performance_score1'],
}
PERFORMANCE_NAMES = [
    'Probabilistic\nreasoning', 'Horizon\ntask', 'Restless\nbandit',
    'Instrumental\nlearning', 'Two-step\ntask', 'Balloon analog\nrisk task',
]


def _build_normalized_df(experiments, metric_names):
    """Return (df, df_cis) with columns = metric_names + 'Agent', normalized
    so that Random=0 and Human average=1."""
    llm_df = pd.read_csv('./data/llm_features.csv')
    metrics, metrics_cis = merge_all_metrics_and_features(
        experiments, _exclude_agents(), llm_df,
    )

    df = pd.DataFrame(metrics).T
    df_cis = pd.DataFrame(metrics_cis).T
    df.columns = metric_names
    df_cis.columns = metric_names
    df = df.reset_index().rename(columns={'index': 'Agent'})
    df_cis = df_cis.reset_index().rename(columns={'index': 'Agent'})

    # Normalize: shift by random, scale by human
    for m in metric_names:
        rand_val = df.loc[df['Agent'] == 'random', m].values
        df[m] = abs(df[m] - rand_val)
        human_val = df.loc[df['Agent'] == 'human', m].values
        df[m] = df[m] / human_val
        df_cis[m] = abs(df_cis[m]) / human_val

    return df, df_cis


# ── grouped vertical bar chart ─────────────────────────────────────────────

def plot_grouped_bars(models, interest, store_id):
    """Produce a grouped vertical bar chart comparing *models* on either
    'performance' or 'behaviour' metrics."""

    if interest == 'behaviour':
        experiments, metric_names = BEHAVIOUR_EXPERIMENTS, BEHAVIOUR_NAMES
        ylabel = 'Parameter value'
    else:
        experiments, metric_names = PERFORMANCE_EXPERIMENTS, PERFORMANCE_NAMES
        ylabel = 'Performance'

    df, df_cis = _build_normalized_df(experiments, metric_names)

    # Keep only requested models (drop human/random rows)
    present = [m for m in models if m in df['Agent'].values]
    if not present:
        print(f'[grouped_comparison] None of {models} found in scores. Skipping {interest} plot.')
        return
    missing = set(models) - set(present)
    if missing:
        print(f'[grouped_comparison] Warning: engines not found in scores: {missing}')

    n_metrics = len(metric_names)
    n_models = len(present)
    x = np.arange(n_metrics)
    width = 0.8 / n_models  # bar width

    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(max(7, n_metrics * 0.9), 4))

    for i, model in enumerate(present):
        row = df[df['Agent'] == model]
        row_ci = df_cis[df_cis['Agent'] == model]
        vals = [row[m].values[0] for m in metric_names]
        cis = [row_ci[m].values[0] / 2 for m in metric_names]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, yerr=cis, label=model,
               color=palette[i % len(palette)], alpha=0.75,
               capsize=3, edgecolor='white', linewidth=0.5)

    # Reference lines
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, label='Humans')
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, label='Random')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=8, ha='center')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    os.makedirs('./plots/grouped', exist_ok=True)
    outpath = f'./plots/grouped/{store_id}_{interest}.pdf'
    plt.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# ── MDS embedding plot ──────────────────────────────────────────────────────

def plot_embedding(models, store_id):
    """Produce a 2-D MDS embedding of the 10 behavioural metrics for all
    available engines, highlighting *models* with colour and labelling them."""

    metric_names = BEHAVIOUR_NAMES
    llm_df = pd.read_csv('./data/llm_features.csv')
    metrics, _ = merge_all_metrics_and_features(
        BEHAVIOUR_EXPERIMENTS, _exclude_agents(), llm_df,
    )

    agents = list(metrics.keys())
    X = np.array(list(metrics.values()))

    # Standardise before MDS
    X = StandardScaler().fit_transform(X)

    mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
    embedding = mds.fit_transform(X)

    # Assign colours: highlighted models get distinct colours, rest is grey
    palette = plt.cm.tab10.colors
    highlight_set = set(models) | {'human', 'random'}
    # Build ordered colour list
    highlight_list = [a for a in agents if a in highlight_set]
    color_map = {}
    ci = 0
    for a in highlight_list:
        if a == 'human':
            color_map[a] = 'black'
        elif a == 'random':
            color_map[a] = 'grey'
        else:
            color_map[a] = palette[ci % len(palette)]
            ci += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, agent in enumerate(agents):
        c = color_map.get(agent, '#cccccc')
        s = 60 if agent in highlight_set else 20
        zorder = 3 if agent in highlight_set else 1
        ax.scatter(embedding[i, 0], embedding[i, 1], c=[c], s=s, zorder=zorder)
        if agent in highlight_set:
            ax.annotate(agent, (embedding[i, 0], embedding[i, 1]),
                        fontsize=8, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Embedding dimension 1', fontsize=10)
    ax.set_ylabel('Embedding dimension 2', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    os.makedirs('./plots/grouped', exist_ok=True)
    outpath = f'./plots/grouped/{store_id}_embedding.pdf'
    plt.savefig(outpath)
    plt.close()
    print(f'Saved {outpath}')


# ── CLI ─────────────────────────────────────────────────────────────────────

def run(models=None, interest=None, embedding=None, store_id=None):
    parser = argparse.ArgumentParser(
        description='Grouped bar charts & MDS embedding in the style of the Centaur paper.')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Engine names to compare (e.g. socius/Llama-Centaur-1B unsloth/Llama-3.2-1B).')
    parser.add_argument('--interest', choices=['behaviour', 'performance', 'both'],
                        default='both',
                        help='Which grouped bar chart(s) to produce.')
    parser.add_argument('--embedding', action='store_true',
                        help='Also produce the MDS embedding plot.')
    parser.add_argument('--store_id', type=str, default='comparison',
                        help='Prefix for output filenames.')
    args = parser.parse_args()

    if models is None:
        models = args.models
    if interest is None:
        interest = args.interest
    if embedding is None:
        embedding = args.embedding
    if store_id is None:
        store_id = args.store_id

    if interest in ('performance', 'both'):
        plot_grouped_bars(models, 'performance', store_id)
    if interest in ('behaviour', 'both'):
        plot_grouped_bars(models, 'behaviour', store_id)
    if embedding:
        plot_embedding(models, store_id)


if __name__ == '__main__':
    run()
