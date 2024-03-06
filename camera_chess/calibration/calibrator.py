import os

import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def platt_scaling_logits(p, bias, conf_weight):
    p = torch.tensor(p, dtype=torch.float)
    p = torch.logit(p)
    res = 1 / (1 + torch.exp(bias + conf_weight * p))
    return res


class Calibrator:
    def __init__(self, match_df, n_bins):
        self.match_df = match_df
        self.n_bins = n_bins

        self.attrs = [col.replace('pred_', '') for col in self.match_df.columns if col.startswith('pred_')]
        self.preds_and_target = self._get_preds_and_target()

    @staticmethod
    def compute_metric(stats):
        metric = np.sum(stats['counts'] * np.abs(stats['metrics'] - stats['confidences'])) / np.sum(stats['counts'])
        return metric

    @staticmethod
    def _reliability_diagram_subplot(ax, bin_data, title="Reliability Diagram"):
        confidences = bin_data["confidences"]
        counts = bin_data["counts"]
        bins = bin_data["bins"]

        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0

        min_count = np.min(counts)
        max_count = np.max(counts)
        normalized_counts = (counts - min_count) / (max_count - min_count)

        alphas = 0.2 + 0.8 * normalized_counts
        widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts

        colors = np.zeros((len(counts), 4), dtype=np.float32)
        colors[:, 0] = 240 / 255
        colors[:, 1] = 60 / 255
        colors[:, 2] = 60 / 255
        colors[:, 3] = alphas

        gap_plt = ax.bar(positions, np.abs(bin_data['metrics'] - confidences),
                         bottom=np.minimum(bin_data['metrics'], confidences), width=widths,
                         edgecolor=colors, color=colors, linewidth=1, label="Gap")
        metric_plt = ax.bar(positions, 0, bottom=bin_data['metrics'], width=widths,
                            edgecolor="black", color="black", alpha=1.0, linewidth=3,
                            label="Precision")

        ax.set_aspect("equal")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(handles=[gap_plt, metric_plt])

    @staticmethod
    def _confidence_histogram_subplot(ax, bin_data, title="Examples per bin"):
        metric_plt = ax.axvline(x=bin_data['avg_metric'], ls="solid", lw=3, c="black",
                                label='Precision')
        conf_plt = ax.axvline(x=bin_data['avg_confidence'], ls="dotted", lw=3,
                              c="#444", label='Avg. confidence')

        counts = bin_data["counts"]
        bins = bin_data["bins"]
        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0
        ax.bar(positions, counts, width=bin_size * 0.9)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.legend(handles=[metric_plt, conf_plt])

    def _get_preds_and_target(self):
        preds_and_target = {}
        for attr in self.attrs:
            preds = np.clip(self.match_df[f'pred_{attr}'].values, a_min=0.0001, a_max=0.9999)
            target = self.match_df[f'gt_{attr}'].values.astype(np.int32)

            annotated_idx = target > -1
            preds = preds[annotated_idx]
            target = target[annotated_idx]

            preds_and_target[attr] = {
                'preds': preds,
                'target': target
            }

        return preds_and_target

    def _plot(self, bin_data, title, figsize, dpi):
        figsize = (figsize[0], figsize[0] * 1.4)

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=figsize, dpi=dpi,
                               gridspec_kw={"height_ratios": [8, 2]})
        plt.tight_layout(pad=1.25)

        self._reliability_diagram_subplot(ax[0], bin_data, title=title)

        # Draw the confidence histogram upside down
        orig_counts = bin_data["counts"]
        bin_data["counts"] = -bin_data["counts"]
        self._confidence_histogram_subplot(ax[1], bin_data, title="")
        bin_data["counts"] = orig_counts

        # Also negate the ticks for the upside-down histogram
        y_ticks = ax[1].get_yticks()
        new_ticks = np.abs(y_ticks).astype(int)
        ax[1].set_yticks(y_ticks)
        ax[1].set_yticklabels(new_ticks)

        return fig

    def _compute_stats(self, true_labels, confidences):
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        indices = np.digitize(confidences, bins, right=True)

        bin_confidences = np.zeros(self.n_bins, dtype=float)
        bin_metrics = np.zeros(self.n_bins, dtype=float)
        bin_counts = np.zeros(self.n_bins, dtype=int)

        for b in range(self.n_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) == 0:
                continue

            bin_metrics[b] = np.mean(true_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

        avg_metric = np.sum(bin_metrics * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        res = {
            "metrics": bin_metrics,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_metric": avg_metric,
            "avg_confidence": avg_conf
        }

        return res

    def minimize_log_loss(self, map_function, x0):
        results = {}
        for attr, d in self.preds_and_target.items():
            preds = d['preds'].reshape(-1, 1)
            target = torch.tensor(d['target'].reshape(-1, 1), dtype=torch.float)

            def f(x):
                bias = x[0]
                conf_weight = x[1]
                scaled_preds = map_function(preds, bias=bias, conf_weight=conf_weight)

                try:
                    loss = F.binary_cross_entropy(scaled_preds, target).item()
                except RuntimeError:
                    return 100

                return loss

            best_score = float('inf')
            best_x = None
            for method in ['BFGS', 'Nelder-Mead', 'Powell', 'CG', 'L-BFGS-B']:
                res = minimize(f, x0=x0, method=method)
                score = f(res.x)
                if score < best_score:
                    best_score = score
                    best_x = {'bias': res.x[0], 'conf_weight': res.x[1]}

            ic(attr, best_score, best_x)
            results[attr] = {"params": best_x}

        return results

    def generate_plots(self, plots_dir, suffix=""):
        os.makedirs(plots_dir, exist_ok=True)
        attr_to_stats = {}
        for attr, d in self.preds_and_target.items():
            stats = self._compute_stats(d['target'], d['preds'])
            fig = self._plot(stats, attr, figsize=(12, 12), dpi=72)
            plt.savefig(os.path.join(plots_dir, f'{attr}{suffix}.jpg'))
            plt.close(fig)

            attr_to_stats[attr] = stats

        return attr_to_stats
