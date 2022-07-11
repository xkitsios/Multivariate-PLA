import csv
import time

import numpy as np
import pandas as pd

from operator import itemgetter
from intervaltree import IntervalTree, Interval


def quantization(b, e):
    return int(b / e) * e


def merge_data(current_reduced_data, new_data):
    return current_reduced_data.union(new_data)


def swing_filter(ts, e):
    points = []
    a_max = a_min = None
    interval_trees = {}
    wedges_of_b = set()
    for idx in range(len(ts)):
        up_val = ts.iloc[idx] + e
        down_val = ts.iloc[idx] - e

        if len(points) == 0:
            points = [idx]
            continue

        if len(points) >= 2:
            up_lim = a_max * (idx - points[0]) + quantization(ts.iloc[points[0]], e)
            down_lim = a_min * (idx - points[0]) + quantization(ts.iloc[points[0]], e)
            if (not np.isclose(down_val, up_lim) and down_val > up_lim) or (
                    not np.isclose(up_val, down_lim) and up_val < down_lim):
                b = quantization(ts.iloc[points[0]], e)
                if b not in interval_trees:
                    interval_trees[b] = IntervalTree()
                tree = interval_trees[b]
                tree.addi(a_min, a_max, {points[0]})
                points = [idx]
                continue

        a_max_temp = (up_val - quantization(ts.iloc[points[0]], e)) / (idx - points[0])
        a_min_temp = (down_val - quantization(ts.iloc[points[0]], e)) / (idx - points[0])

        if len(points) == 1:
            a_max = a_max_temp
            a_min = a_min_temp
        else:
            up_lim = a_max * (idx - points[0]) + quantization(ts.iloc[points[0]], e)
            if not np.isclose(up_val, up_lim) and up_val < up_lim:
                a_max = a_max_temp
            down_lim = a_min * (idx - points[0]) + quantization(ts.iloc[points[0]], e)
            if not np.isclose(down_val, down_lim) and down_val > down_lim:
                a_min = a_min_temp

        points.append(idx)

    if len(points) >= 2:
        b = quantization(ts.iloc[points[0]], e)
        if b not in interval_trees:
            interval_trees[b] = IntervalTree()
        tree = interval_trees[b]
        tree.addi(a_min, a_max, {points[0]})

    # for tree in interval_trees.values():
    #     tree.merge_equals(data_reducer=merge_data)

    return interval_trees, ts.index[-1]


def reconstruct(interval_trees, last_idx):
    lines = []
    for b, trees in interval_trees.items():
        for interval in trees:
            for t_start in interval.data:
                lines.append([t_start, interval.begin, interval.end, b])
    lines = sorted(lines, key=itemgetter(0))

    x = []
    y = []
    for line_idx in range(len(lines)):
        if line_idx + 1 == len(lines):
            t_end = last_idx + 1
        else:
            t_end = lines[line_idx + 1][0]
        t_start, a_max, a_min, b = lines[line_idx][0], lines[line_idx][1], lines[line_idx][2], lines[line_idx][3]
        for t in range(t_start, t_end):
            x.append(t)
            y_val = ((a_max + a_min) / 2) * (t - t_start) + b
            y.append(y_val)

    return pd.Series(y, index=x)


def validate(original, approximation, e):
    cmp = original.compare(approximation, keep_equal=True)
    cmp['valid'] = np.isclose(cmp['self'], cmp['other'], atol=e)
    num_errors = len(cmp) - cmp['valid'].sum()

    if num_errors > 0:
        print('Found', num_errors, 'error')
        print(cmp[cmp['valid'] == False])
        return False
    else:
        return True


def merge_intervals(interval_trees):
    for b, tree in interval_trees.items():
        changed = True
        while changed:
            changed = False
            intervals = set(tree)
            for interval in intervals:
                overlap_intervals = tree.overlap(interval)
                if len(overlap_intervals) < 2:
                    continue
                new_interval = overlap_intervals.pop()
                tree.remove(new_interval)
                for overlap_interval in overlap_intervals:
                    if not new_interval.overlaps(overlap_interval):
                        continue
                    new_interval = Interval(
                        max(new_interval.begin, overlap_interval.begin),
                        min(new_interval.end, overlap_interval.end),
                        new_interval.data.union(overlap_interval.data)
                    )
                    tree.remove(overlap_interval)
                tree.add(new_interval)
                changed = True


def merge_intervals_old(interval_trees):
    for b, tree in interval_trees.items():
        new_tree = IntervalTree()
        to_be_removed = set()
        for interval in tree:
            overlap_intervals = tree.overlap(interval)
            if len(overlap_intervals) == 1:
                new_tree.add(interval)
                to_be_removed.add(interval)
        for interval in to_be_removed:
            tree.remove(interval)
        change = True
        while change:
            change = False
            to_be_removed = set()
            results = set()
            for interval in tree:
                overlap_intervals = tree.overlap(interval)
                if len(overlap_intervals) < 2:
                    new_tree.add(interval)
                    continue
                new_interval = interval.copy()
                for overlap_interval in overlap_intervals:
                    if not new_interval.overlaps(overlap_interval):
                        continue
                    # print('Interval 1', new_interval.begin, new_interval.end)
                    # print('Interval 2', overlap_interval.begin, overlap_interval.end)
                    new_interval = Interval(
                        max(new_interval.begin, overlap_interval.begin),
                        min(new_interval.end, overlap_interval.end),
                        new_interval.data.union(overlap_interval.data)
                    )
                    # print('Interval merged', new_interval.begin, new_interval.end)
                    if new_interval.begin > new_interval.end:
                        raise Exception
                    to_be_removed.add(overlap_interval)
                new_tree.add(new_interval)
                for item in to_be_removed:
                    tree.remove(item)
                change = True
                break
        tree |= new_tree


def main(filenames, e_vals, multivariate=True, univariate=True):
    all_ts = {}
    swing_lines = {}
    output = []
    print('Applying Swing Filter')
    for f in filenames:
        print(f)
        df = pd.read_csv(f)
        ts = pd.Series(df['value'], index=df.index)
        all_ts[f] = ts
        swing_lines[f] = {}
        for e in e_vals:
            print('e:', e)
            interval_trees, last_idx = swing_filter(ts, e)
            ts_reconstructed = reconstruct(interval_trees, last_idx)
            validate(ts, ts_reconstructed, e)
            swing_lines[f][e] = [interval_trees, last_idx]
            print('Completed!')

    if univariate:
        print('Univariate Merging')
        for e in e_vals:
            print('e:', e)
            for f in filenames:
                print(f)
                ts = all_ts[f]
                [interval_trees, last_idx] = swing_lines[f][e]
                segments = 0
                for tree in interval_trees.values():
                    segments += len(tree)
                start = time.time()
                for b, tree in interval_trees.items():
                    tree.merge_overlaps(data_reducer=merge_data)
                # merge_intervals(interval_trees)
                print('Merge Time: ', time.time() - start)
                ts_reconstructed = reconstruct(interval_trees, last_idx)
                validate(ts, ts_reconstructed, e)
                segments_merged = 0
                for tree in interval_trees.values():
                    segments_merged += len(tree)

                print('Merged!')
                print('Total Segments: ', segments)
                print('Total Segments after merge: ', segments_merged)
                print('Diff: ', segments - segments_merged)
                output.append([f, ts.shape[0], e, segments_merged, segments - segments_merged, segments])

    if multivariate:
        print('Multivariate Merging')
        for e in e_vals:
            print('e:', e)
            lines_counter = 0
            for f in filenames:
                print(f)
                ts = all_ts[f]
                [interval_trees, last_idx] = swing_lines[f][e]
                lines_counter += len(interval_trees)
                merge_intervals(interval_trees)
                ts_reconstructed = reconstruct(interval_trees, last_idx)
                validate(ts, ts_reconstructed, e)

            print('Merged!')
            print('Total Segments: ', lines_counter)
            # print('Segments with Lines: ', len(wedges))
            # print('Segments with Refs: ', lines_counter - len(wedges))

    with open('output.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(output)


if __name__ == "__main__":
    files = [#'paris_scaled.csv']
             # 'dataset/AristotelousCleanTempScaled.csv',
             # 'dataset/AthensCleanTempScaled.csv',
             # 'dataset/ElefsinaCleanTempScaled.csv',
             # 'dataset/GeoponikiCleanTempScaled.csv',
             # 'dataset/KoropiCleanTempScaled.csv',
             # 'dataset/LiosiaCleanTempScaled.csv',
             # 'dataset/LykovrisiCleanTempScaled.csv',
             # 'dataset/MarousiCleanTempScaled.csv',
             # 'dataset/NeaSmirniCleanTempScaled.csv',
             # 'dataset/PatisionCleanTempScaled.csv',
             # 'dataset/PeristeriCleanTempScaled.csv',
             # 'dataset/PireusCleanTempScaled.csv',
              'dataset/ThrakomakedonesCleanTempScaled.csv']

    main(files, [0.025], multivariate=False, univariate=True)
