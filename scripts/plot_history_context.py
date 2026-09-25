"""Plot a downloaded history_context run without loading checkpoints."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    if report['status'] != 'COMPLETE':
        raise ValueError('Only completed runs can produce the presentation figures')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scripts.analyze_world_model_capabilities import MODELS, LABELS
    colors = plt.get_cmap('tab20').colors
    coordinates = report['models'][0]['physical_coordinates']
    horizons = report['settings']['horizons']
    with PdfPages(args.run / 'history_effect.pdf') as pdf:
        for cohort in ('all', 'uniform', 'motion'):
            fig, axes = plt.subplots(1 + len(horizons), len(coordinates),
                                     figsize=(4 * len(coordinates), 3 * (1 + len(horizons))), squeeze=False)
            for result in report['models']:
                i = MODELS.index(result['model'])
                contexts = sorted(map(int, result['conditions']))
                for row, (metric, horizon) in enumerate([('current', 0), *[('forecast', h) for h in horizons]]):
                    for col, coordinate in enumerate(coordinates):
                        ax = axes[row, col]
                        values = [result['conditions'][str(c)][cohort][metric][str(horizon)][coordinate] for c in contexts]
                        ax.plot(contexts, values, marker='o', color=colors[i], label=LABELS[i])
                        ax.set(xscale='log', yscale='log', xticks=contexts,
                               title=f'{metric}, step {horizon}: {coordinate}', xlabel='Observed frames',
                               ylabel=f"RMSE [{result['physical_units'][coordinate]}]")
                        ax.set_xticklabels(contexts)
                        ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=7)
            fig.suptitle(f"Frozen Cartpole checkpoints: {cohort} windows | {report['run_name']}")
            fig.tight_layout(rect=(0, .06, 1, .97))
            pdf.savefig(fig)
            fig.savefig(args.run / f'history_effect_{cohort}.png', dpi=160)
            plt.close(fig)
    print(f"Run: {report['run_name']} | Plots: {args.run / 'history_effect.pdf'}")


if __name__ == '__main__':
    main()
