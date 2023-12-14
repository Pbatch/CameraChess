import argparse
import io
import json
import os

import chess
import pandas as pd
import seaborn as sns
from icecream import ic
import datetime as dt

from camera_chess.constants import DATA_DIR
from camera_chess.sequence_generator import SequenceGenerator
from camera_chess.tracker import Tracker
import matplotlib.pyplot as plt


def load_scores(scores_path):
    with open(scores_path, 'r') as f:
        scores = json.load(f)
    return scores


def create_scores(scores_path, model_basename, force):
    datasets = [*[os.path.join('youtube', s) for s in ['carlsen_vidit', 'anand_carlsen', 'dubov_nepo', 'carlsen_abdu',
                                                       'carlsen_toma', 'duda_carlsen', 'gukesh_shakh', 'hans_rinat',
                                                       'hari_tuan', 'harika_nana', 'hikaru_sarin', 'hikaru_vasif',
                                                       'magnus_madaminov', 'ramirez_yoo', 'shimanov_vidit', 'cramling_ukraine',
                                                       'cramling_boris']],
                *[os.path.join('mercato', s) for s in ['bogdan', 'elephant', 'english', 'james', 'reti',
                                                       'slav', 'ruy', 'vienna']],
                *[os.path.join('four_corners', s) for s in ['caro', 'french', 'london', 'ponziani', 'tromp']],
                *[os.path.join('tom', s) for s in ['slav', 'spanish', 'italian']]]

    scores = {}
    for dataset in datasets:
        sequence_generator = SequenceGenerator(dataset, model_basename)
        tracker = Tracker(dataset, sequence_generator.model_basename)
        sequence_generator.create_sequence(force=force)
        logs = tracker.process_sequence(sequence_generator.sequence_path, force=force)

        pred_pgn = logs[max(logs.keys(), key=lambda x: int(x))]['pgn']
        pred_moves = [str(move) for move in chess.pgn.read_game(io.StringIO(pred_pgn)).mainline_moves()]

        gt_moves = sequence_generator.video_config.moves

        score = 0
        pred_fail = None
        gt_fail = None
        halfmove_fail = -1
        board = chess.Board()
        for i in range(len(gt_moves)):
            pred_move = pred_moves[i] if i < len(pred_moves) else None
            gt_move = gt_moves[i]
            if pred_move == gt_move:
                score += 1
                board.push(board.parse_uci(gt_move))
            else:
                pred_fail = board.san(board.parse_uci(pred_move)) if pred_move is not None else ""
                gt_fail = board.san(board.parse_uci(gt_move))
                halfmove_fail = i
                break
        score = round(100 * score / len(gt_moves))
        scores[dataset] = {'score': score,
                           'halfmoves': len(gt_moves),
                           'pred_fail': pred_fail,
                           'gt_fail': gt_fail,
                           'halfmove_fail': halfmove_fail
                           }
        ic(dataset, scores[dataset])

    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)

    return scores


def plot_scores(scores, plot_path):
    data = []
    for k, v in scores.items():
        dataset = k.split(os.path.sep, 1)[0]
        score = v['score']
        data.append([dataset, score])

    df = pd.DataFrame(data, columns=['dataset', 'score'])
    score_dict = df.groupby("dataset")['score'].mean().to_dict()
    score_string = " | ".join([f'{k}:{round(v)}' for k, v in score_dict.items()])

    title = f'Tracking scores ({score_string})'
    sns.histplot(data=df, x="score", hue="dataset", multiple="dodge", bins=20).set(title=title)

    plt.savefig(plot_path)


def main(model_basename, force):
    run_id = f"{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{model_basename.split('.')[0]}"
    scores_path = os.path.join(DATA_DIR, f'{run_id}_scores.json')
    plot_path = os.path.join(DATA_DIR, f'{run_id}_plot.jpg')
    scores = create_scores(scores_path, model_basename, force)

    # scores_path = os.path.join(DATA_DIR, "2023-10-09-16-29-57_scores.json")
    # plot_path = os.path.join(DATA_DIR, "2023-10-09-16-29-57_plot.jpg")
    # scores = load_scores(scores_path)

    plot_scores(scores, plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_basename', '-m', type=str, required=True)
    parser.add_argument('--force', '-f', action="store_true")
    args = parser.parse_args()
    main(args.model_basename, args.force)
