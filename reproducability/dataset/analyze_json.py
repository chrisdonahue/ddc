if __name__ == '__main__':
    import argparse
    from collections import Counter, defaultdict
    import json
    import os
    from util import get_subdirs

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_fps', type=str, nargs='+', help='List of dataset filepaths to analyze')
    parser.add_argument('--diff', type=str, help='If provided, only analyze charts of this difficulty')

    parser.set_defaults(
        diff='')

    args = parser.parse_args()

    json_fps = []
    for dataset_fp in args.dataset_fps:
        with open(dataset_fp, 'r') as f:
            json_fps += f.read().splitlines()

    chart_types = Counter()
    chart_diff_coarse = Counter()
    chart_feet = Counter()
    chart_coarse_to_stream = {}
    chart_freetexts = Counter()
    beat_phases = Counter()
    vocab = Counter()
    songs_time_annotated = 0.0
    charts_time_annotated = 0.0
    stream_total = 0.0
    feet_total = 0.0
    arrows_total = 0
    chart_coarse_to_superset = defaultdict(list)
    for json_fp in json_fps:
        with open(json_fp, 'r') as f:
            song_meta = json.loads(f.read())

        max_time_annotated = -1.0
        coarse_to_beats = defaultdict(set)
        for chart_meta in song_meta['charts']:
            if args.diff and chart_meta['difficulty_coarse'] != args.diff:
                continue

            coarse = chart_meta['difficulty_coarse']
            feet = chart_meta['difficulty_fine']
            feet_total += feet

            chart_types[chart_meta['type']] += 1
            chart_diff_coarse[coarse] += 1
            chart_feet[feet] += 1
            chart_freetexts[chart_meta['desc_or_author']] += 1

            num_arrows = 0
            for _, beat, time, arrow in chart_meta['notes']:
                beat_phase = beat - int(beat)
                beat_phase = int(beat_phase * 100.0) / 100.0
                beat_phases[beat_phase] += 1
                vocab[arrow] += 1
                if arrow != '0' * len(arrow):
                    num_arrows += 1
                coarse_to_beats[coarse].add(beat)
            arrows_total += num_arrows

            chart_time_annotated = chart_meta['notes'][-1][2] - chart_meta['notes'][0][2]
            if chart_time_annotated > max_time_annotated:
                max_time_annotated = chart_time_annotated
            charts_time_annotated += chart_time_annotated

            stream = num_arrows / chart_time_annotated
            stream_total += stream
            if feet not in chart_coarse_to_stream:
                chart_coarse_to_stream[coarse] = []
            chart_coarse_to_stream[coarse].append(stream)

        songs_time_annotated += max_time_annotated

        coarses = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
        for i, coarse in enumerate(coarses):
            for coarse_next in coarses:
                beats = coarse_to_beats[coarse]
                beats_next = coarse_to_beats[coarse_next]
                chart_coarse_to_superset[(coarse, coarse_next)].append(len(beats & beats_next) / float(len(beats)))

    chart_coarse_to_stream = {k: sum(l) / len(l) for k, l in chart_coarse_to_stream.items()}
    chart_coarse_to_superset = {k: (reduce(lambda x, y: x + y, l) / len(l)) for k, l in chart_coarse_to_superset.items()}

    nsongs = len(json_fps)
    ncharts = sum(chart_feet.values())
    print ','.join(args.dataset_fps)
    print 'Num songs: {}'.format(nsongs)
    print 'Total music annotated (s): {}'.format(songs_time_annotated)
    print 'Avg song length (s): {}'.format(songs_time_annotated / nsongs)

    print 'Num charts: {}'.format(ncharts)
    print 'Avg num charts per song: {}'.format(float(ncharts) / nsongs)
    print 'Total chart time annotated (s): {}'.format(charts_time_annotated)
    print 'Avg chart length (s): {}'.format(charts_time_annotated / ncharts)
    print 'Avg chart length (steps): {}'.format(float(arrows_total) / ncharts)

    print 'Chart types: {}'.format(chart_types)
    print 'Chart coarse difficulties: {}'.format(chart_diff_coarse)
    print 'Chart feet: {}'.format(chart_feet)
    print 'Chart coarse avg arrows per second: {}'.format(chart_coarse_to_stream)
    print 'Chart coarse avg superset: {}'.format(chart_coarse_to_superset)
    print 'Chart freetext fields: {}'.format(chart_freetexts)
    print 'Chart vocabulary (size={}): {}'.format(len(vocab), vocab)
    print 'Beat phases: {}'.format(beat_phases)

    print 'Avg feet: {}'.format(feet_total / ncharts)
    print 'Avg arrows per second: {}'.format(stream_total / ncharts)
