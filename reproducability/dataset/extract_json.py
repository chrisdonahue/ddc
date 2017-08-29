import glob
import logging as smlog
import os
import traceback

from smdataset.abstime import calc_note_beats_and_abs_times
from smdataset.parse import parse_sm_txt

_ATTR_REQUIRED = ['offset', 'bpms', 'notes']

if __name__ == '__main__':
    import argparse
    from collections import OrderedDict
    import json
    json.encoder.FLOAT_REPR = lambda f: ('%.6f' % f)
    from util import ez_name, get_subdirs

    parser = argparse.ArgumentParser()
    parser.add_argument('packs_dir', type=str, help='Directory of packs (organized like Stepmania songs folder)')
    parser.add_argument('json_dir', type=str, help='Output JSON directory')
    parser.add_argument('--itg', dest='itg', action='store_true', help='If set, subtract 9ms from offset')
    parser.add_argument('--choose', dest='choose', action='store_true', help='If set, choose from list of packs')

    parser.set_defaults(
        itg=False,
        choose=False)

    args = parser.parse_args()

    pack_names = get_subdirs(args.packs_dir, args.choose)
    pack_dirs = [os.path.join(args.packs_dir, pack_name) for pack_name in pack_names]
    pack_sm_globs = [os.path.join(pack_dir, '*', '*.sm') for pack_dir in pack_dirs]

    if not os.path.isdir(args.json_dir):
        os.mkdir(args.json_dir)

    pack_eznames = set()
    for pack_name, pack_sm_glob in zip(pack_names, pack_sm_globs):
        pack_sm_fps = sorted(glob.glob(pack_sm_glob))
        pack_ezname = ez_name(pack_name)
        if pack_ezname in pack_eznames:
            raise ValueError('Pack name conflict: {}'.format(pack_ezname))
        pack_eznames.add(pack_ezname)

        if len(pack_sm_fps) > 0:
            pack_outdir = os.path.join(args.json_dir, pack_ezname)
            if not os.path.isdir(pack_outdir):
                os.mkdir(pack_outdir)

        sm_eznames = set()
        for sm_fp in pack_sm_fps:
            sm_name = os.path.split(os.path.split(sm_fp)[0])[1]
            sm_ezname = ez_name(sm_name)
            if sm_ezname in sm_eznames:
                raise ValueError('Song name conflict: {}'.format(sm_ezname))
            sm_eznames.add(sm_ezname)

            with open(sm_fp, 'r') as sm_f:
                sm_txt = sm_f.read()

            # parse file
            try:
                sm_attrs = parse_sm_txt(sm_txt)
            except ValueError as e:
                smlog.error('{} in\n{}'.format(e, sm_fp))
                continue
            except Exception as e:
                smlog.critical('Unhandled parse exception {}'.format(traceback.format_exc()))
                raise e

            # check required attrs
            try:
                for attr_name in _ATTR_REQUIRED:
                    if attr_name not in sm_attrs:
                        raise ValueError('Missing required attribute {}'.format(attr_name))
            except ValueError as e:
                smlog.error('{}'.format(e))
                continue

            # handle missing music
            root = os.path.abspath(os.path.join(sm_fp, '..'))
            music_fp = os.path.join(root, sm_attrs.get('music', ''))
            if 'music' not in sm_attrs or not os.path.exists(music_fp):
                music_names = []
                sm_prefix = os.path.splitext(sm_name)[0]

                # check directory files for reasonable substitutes
                for filename in os.listdir(root):
                    prefix, ext = os.path.splitext(filename)
                    if ext.lower()[1:] in ['mp3', 'ogg']:
                        music_names.append(filename)

                try:
                    # handle errors
                    if len(music_names) == 0:
                        raise ValueError('No music files found')
                    elif len(music_names) == 1:
                        sm_attrs['music'] = music_names[0]
                    else:
                        raise ValueError('Multiple music files {} found'.format(music_names))
                except ValueError as e:
                    smlog.error('{}'.format(e))
                    continue

                music_fp = os.path.join(root, sm_attrs['music'])

            bpms = sm_attrs['bpms']
            offset = sm_attrs['offset']
            if args.itg:
                # Many charters add 9ms of delay to their stepfiles to account for ITG r21/r23 global delay
                # see http://r21freak.com/phpbb3/viewtopic.php?f=38&t=12750
                offset -= 0.009
            stops = sm_attrs.get('stops', [])

            out_json_fp = os.path.join(pack_outdir, '{}_{}.json'.format(pack_ezname, sm_ezname))
            out_json = OrderedDict([
                ('sm_fp', os.path.abspath(sm_fp)),
                ('music_fp', os.path.abspath(music_fp)),
                ('pack', pack_name),
                ('title', sm_attrs.get('title')),
                ('artist', sm_attrs.get('artist')),
                ('offset', offset),
                ('bpms', bpms),
                ('stops', stops),
                ('charts', [])
            ])

            for idx, sm_notes in enumerate(sm_attrs['notes']):
                note_beats_and_abs_times = calc_note_beats_and_abs_times(offset, bpms, stops, sm_notes[5])
                notes = {
                    'type': sm_notes[0],
                    'desc_or_author': sm_notes[1],
                    'difficulty_coarse': sm_notes[2],
                    'difficulty_fine': sm_notes[3],
                    'notes': note_beats_and_abs_times,
                }
                out_json['charts'].append(notes)

            with open(out_json_fp, 'w') as out_f:
                try:
                    out_f.write(json.dumps(out_json))
                except UnicodeDecodeError:
                    smlog.error('Unicode error in {}'.format(sm_fp))
                    continue

            print 'Parsed {} - {}: {} charts'.format(pack_name, sm_name, len(out_json['charts']))
