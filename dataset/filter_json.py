if __name__ == '__main__':
    import argparse
    import copy
    import json
    import os
    from util import get_subdirs

    parser = argparse.ArgumentParser()
    parser.add_argument('json_in_dir', type=str, help='Input JSON directory')
    parser.add_argument('json_out_dir', type=str, help='Output (filtered) JSON directory')
    parser.add_argument('--chart_types', type=str, help='Whitelist of chart types; if empty, no filter')
    parser.add_argument('--chart_difficulties', type=str, help='Whitelist of chart difficulties; if empty, no filter')
    parser.add_argument('--min_chart_feet', type=int, help='Min chart feet; if negative, no min')
    parser.add_argument('--max_chart_feet', type=int, help='Max chart feet; if negative, no max')
    parser.add_argument('--substitutions', type=str, help='CSV pairs of arrow type substitutions')
    parser.add_argument('--arrow_types', type=str, help='CSV whitelist of arrow types; \'0\' included by default; if empty, no filter')
    parser.add_argument('--max_jump_size', type=int, help='Maximum number of simultaneous arrows allowed; if negative, no max')
    parser.add_argument('--remove_zeros', dest='remove_zeros', action='store_true', help='If set, removes all noops (e.g. 0000) from annotations')
    parser.add_argument('--keep_empty', dest='keep_empty', action='store_true', help='If set, JSON for songs with all charts filtered will be preserved')
    parser.add_argument('--reduce_ppms', dest='reduce_ppms', action='store_true', help='If set, pulse per measure will be filtered if it was previously overspecified')
    parser.add_argument('--ppms', type=str, help='CSV whitelist of allowable pulses per measure; if empty, no filter')
    parser.add_argument('--permutations', type=str, help='List of permutations to include in output')
    parser.add_argument('--choose', dest='choose', action='store_true', help='If set, choose from list of packs')

    parser.set_defaults(
        chart_types='',
        chart_difficulties='',
        min_chart_feet=-1,
        max_chart_feet=-1,
        substitutions='',
        arrow_types='',
        max_jump_size=-1,
        remove_zeros=False,
        keep_empty=False,
        reduce_ppms=False,
        ppms='',
        permutations='0123',
        choose=False)

    args = parser.parse_args()

    chart_types = set(filter(lambda x: bool(x), [x.strip() for x in args.chart_types.split(',')]))
    chart_difficulties = set(filter(lambda x: bool(x), [x.strip() for x in args.chart_difficulties.split(',')]))
    substitutions = filter(lambda x: bool(x), [x.strip() for x in args.substitutions.split(',')])
    assert len(substitutions) % 2 == 0
    substitutions = [(substitutions[i], substitutions[i + 1]) for i in xrange(0, len(substitutions), 2)]
    substitutions = {x.strip():y.strip() for x, y in substitutions}
    arrow_types = set(filter(lambda x: bool(x), [x.strip() for x in args.arrow_types.split(',')]))
    arrow_types.add('0')
    ppms = set([int(x.strip()) for x in filter(lambda x: bool(x), args.ppms.split(','))])
    permutations = set([x.strip() for x in filter(lambda x: bool(x), args.permutations.split(','))])

    if len(chart_types) > 0:
        print 'Only accepting the following chart types: {}'.format(chart_types)
    if len(chart_difficulties) > 0:
        print 'Only accepting the following chart difficulties: {}'.format(chart_difficulties)
    if args.min_chart_feet > 0:
        print 'Only accepting charts with min feet: {}'.format(args.min_chart_feet)
    if args.max_chart_feet > 0:
        print 'Only accepting charts with max feet: {}'.format(args.max_chart_feet)
    if len(substitutions) > 0:
        print 'Making the following arrow substitutions: {}'.format(substitutions)
    if len(arrow_types) > 0:
        print 'Only accepting the following arrow types: {}'.format(arrow_types)
    if args.max_jump_size > 0:
        print 'Only accepting charts with max jump size: {}'.format(args.max_jump_size)
    if len(args.ppms) > 0:
        print 'Only accepting charts with pulses per measure: {}'.format(args.ppms)

    if not os.path.isdir(args.json_out_dir):
        os.mkdir(args.json_out_dir)

    pack_names = get_subdirs(args.json_in_dir, args.choose)

    for pack_name in pack_names:
        print '-' * 80
        print pack_name

        pack_out_dir = os.path.join(args.json_out_dir, pack_name)
        if not os.path.isdir(pack_out_dir):
            os.mkdir(pack_out_dir)

        pack_dir = os.path.join(args.json_in_dir, pack_name)
        pack_naccepted = 0
        pack_ntotal = 0
        for json_name in os.listdir(pack_dir):
            print '-' * 8
            print json_name

            json_fp = os.path.join(pack_dir, json_name)
            with open(json_fp, 'r') as f:
                song_meta = json.loads(f.read())

            charts_accepted = []
            for chart_meta in song_meta['charts']:
                if len(chart_types) > 0 and chart_meta['type'] not in chart_types:
                    print 'Unacceptable chart type: {}'.format(chart_meta['type'])
                    continue

                if len(chart_difficulties) > 0 and chart_meta['difficulty_coarse'] not in chart_difficulties:
                    print 'Unacceptable chart difficulty: {}'.format(chart_meta['difficulty_coarse'])
                    continue

                if args.min_chart_feet >= 0 and chart_meta['difficulty_fine'] < args.min_chart_feet:
                    print 'Unacceptable chart feet: {}'.format(chart_meta['difficulty_fine'])
                    continue

                if args.max_chart_feet >= 0 and chart_meta['difficulty_fine'] > args.max_chart_feet:
                    print 'Unacceptable chart feet: {}'.format(chart_meta['difficulty_fine'])
                    continue

                if len(substitutions) > 0 or args.remove_zeros:
                    notes_cleaned = []
                    for meas, beat, time, note in chart_meta['notes']:
                        for old, new in substitutions.items():
                            note = note.replace(old, new)

                        if args.remove_zeros and note == '0' * len(note):
                            continue

                        notes_cleaned.append((meas, beat, time, note))
                    chart_meta['notes'] = notes_cleaned

                if len(arrow_types) > 1:
                    bad_types = set()
                    for _, beat, time, note in chart_meta['notes']:
                        for char in note:
                            if char not in arrow_types:
                                bad_types.add(char)
                    if len(bad_types) > 0:
                        print 'Unacceptable chart arrow types: {}'.format(bad_types)
                        continue

                if args.max_jump_size > 0:
                    acceptable = True
                    for _, beat, time, note in chart_meta['notes']:
                        jump_size = 0
                        for char in note:
                            if char != '0':
                                jump_size += 1
                        if jump_size > args.max_jump_size:
                            print 'Unacceptable jump: {}'.format(note)
                            acceptable = False
                            break
                    if not acceptable:
                        continue

                if args.reduce_ppms:
                    import primefac
                    measures = {}
                    for note in chart_meta['notes']:
                        measure = note[0][0]
                        if measure not in measures:
                            measures[measure] = []
                        measures[measure].append(note)
                    measures = [measures[k] for k, v in sorted(measures.items(), key=lambda x: x[0])]

                    notes_cleaned = []
                    for measure in measures:
                        denominator = measure[0][0][1]
                        divisors = list( primefac.primefac(denominator))
                        numerators = [note[0][2] for note in measure]
                        factor = 1
                        for divisor in divisors:
                            if reduce(lambda x, y: x and y, [n % divisor == 0 for n in numerators]):
                                numerators = [n // divisor for n in numerators]
                                factor *= divisor

                        measure_old = measure
                        if factor > 1:
                            measure = [([note[0][0], note[0][1] // factor, note[0][2] // factor], note[1], note[2], note[3]) for note in measure]

                        for note in measure:
                            notes_cleaned.append(note)

                    # TODO: remove when stable
                    assert len(notes_cleaned) == len(chart_meta['notes'])
                    for i, ((measure_num, ppm, p), beat, _, _) in enumerate(notes_cleaned):
                        assert notes_cleaned[i][1] == chart_meta['notes'][i][1]
                        beat_recalc = 4.0 * (measure_num + (p / float(ppm)))
                        assert abs(beat_recalc - beat) < 1e-6

                    chart_meta['notes'] = notes_cleaned

                if len(ppms) > 0:
                    acceptable = True
                    for (_, ppm, _), _, _, _ in chart_meta['notes']:
                        if ppm not in ppms:
                            print 'Unacceptable ppm: {}'.format(ppm)
                            acceptable = False
                            break
                    if not acceptable:
                        continue

                for permutation in permutations:
                    chart_meta_copy = copy.deepcopy(chart_meta)
                    notes_cleaned = []
                    for meas, beat, time, note in chart_meta_copy['notes']:
                        note_new = ''.join([note[int(permutation[i])] for i in xrange(len(permutation))])

                        notes_cleaned.append((meas, beat, time, note_new))
                        chart_meta_copy['notes'] = notes_cleaned

                    charts_accepted.append(chart_meta_copy)

            charts_naccepted = len(charts_accepted)
            charts_ntotal = len(song_meta['charts'])
            pack_naccepted += charts_naccepted
            pack_ntotal += charts_ntotal
            print 'Accepted {}/{}'.format(charts_naccepted, charts_ntotal)

            json_out_fp = os.path.abspath(os.path.join(pack_out_dir, json_name))
            if os.path.isfile(json_out_fp):
                os.remove(json_out_fp)

            if not args.keep_empty and len(charts_accepted) == 0:
                continue

            song_meta['charts'] = charts_accepted
            with open(json_out_fp, 'w') as f:
                f.write(json.dumps(song_meta))
        print 'Pack accepted {}/{}'.format(pack_naccepted, pack_ntotal)
