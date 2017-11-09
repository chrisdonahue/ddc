from collections import OrderedDict
import glob
import json
import logging
import shutil
logging.basicConfig()
logger = logging.getLogger(__name__)

from ddc import BeatTimeCalc
from ddc.util import ezname
from parse import parse_sm_txt

_SM_REQUIRED_ATTRS = ['offset', 'bpms', 'notes']

def parse_pack(pack_dir, out_dir,
    tag=None,
    itg_offset=False,
    delete_pack_dir=False):
  # Create tag directory
  tag_dir = out_dir
  if tag:
    tag_dir = os.path.join(out_dir, tag)
    if not os.path.isdir(tag_dir):
      try:
        os.mkdir(tag_dir)
      except:
        logger.critical('Could not create tag directory {}'.format(tag_dir))
        return False

  # Create pack directory
  pack_ezname = ezname(os.path.split(pack_dir)[1])
  pack_out_dir = os.path.join(tag_dir, pack_ezname)
  if not os.path.isdir(pack_out_dir):
    try:
      os.mkdir(pack_out_dir)
    except:
      logger.critical('Could not create pack directory {}'.format(pack_out_dir))
      return False

  # Find all .SM files in pack
  pack_sm_fps = glob.glob(os.path.join(pack_dir, '*', '*.sm'))
  pack_out_dir = os.path.join(tag_dir, pack_ezname)

  # Parse all .SM files
  pack_song_eznames = set()
  for sm_fp in pack_sm_fps:
    song_dir_name = os.path.split(os.path.split(sm_fp)[0])[1]
    sm_name = os.path.split(os.path.split(sm_fp)[0])[1]
    song_ezname = ezname(song_dir_name)
    if song_ezname in pack_song_eznames:
      logger.critical('Multiple SM files present for name {}'.format(song_ezname))
    pack_song_eznames.add(song_ezname)

    # Read .SM file
    with open(sm_fp, 'r') as sm_f:
      sm_txt = sm_f.read()

    # Parse .SM file to attribute dict
    try:
      sm_attrs = parse_sm_txt(sm_txt)
    except ValueError as e:
      logger.error('{}: Parse error {}'.format(song_ezname, e))
      continue
    except Exception as e:
      logger.critical('{}: Unhandled parse exception {}'.format(song_ezname, traceback.format_exc()))
      continue

    # Check presence of required attrs
    for attr_name in _SM_REQUIRED_ATTRS:
      if attr_name not in sm_attrs:
        logger.error('{}: Missing required attribute {}'.format(song_ezname, attr_name))
        continue

    # Handle missing music
    sm_root = os.path.abspath(os.path.join(sm_fp, '..'))
    music_fp = os.path.join(sm_root, sm_attrs.get('music', ''))
    if 'music' not in sm_attrs or not os.path.exists(music_fp):
      music_names = []
      sm_prefix = os.path.splitext(sm_name)[0]

      # check directory files for reasonable substitutes
      for filename in os.listdir(sm_root):
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
        logger.error('{}'.format(e))
        continue

      music_fp = os.path.join(sm_root, sm_attrs['music'])

    # Copy .SM and audio file to outdir
    identifier = '{}_{}'.format(pack_ezname, song_ezname)
    sm_out_name = '{}.sm'.format(identifier)
    music_out_name = '{}{}'.format(identifier, os.path.splitext(music_fp)[1])
    shutil.copyfile(sm_fp, os.path.join(pack_out_dir, sm_out_name))
    shutil.copyfile(music_fp, os.path.join(pack_out_dir, music_out_name))

    # Retrieve required attrs
    bpms = sm_attrs['bpms']
    offset = sm_attrs['offset']
    if itg_offset:
      # Many charters add 9ms of delay to their stepfiles to account for ITG r21/r23 global delay
      # see http://r21freak.com/phpbb3/viewtopic.php?f=38&t=12750
      offset -= 0.009
    stops = sm_attrs.get('stops', [])

    # Build ordered JSON output dict
    out_json = OrderedDict([
      ('ezid', identifier),
      ('ezpack', pack_ezname),
      ('ezname', song_ezname),
      ('sm_fp', sm_out_name),
      ('audio_fp', music_out_name),
      ('title', sm_attrs.get('title')),
      ('artist', sm_attrs.get('artist')),
      ('offset', offset),
      ('bpms', bpms),
      ('stops', stops),
      ('charts', [])
    ])

    # Calculate chart times
    for idx, sm_notes in enumerate(sm_attrs['notes']):
      btc = BeatTimeCalc(offset, bpms, stops)
      measure_beat_time_steps = []
      for measure_num, measure in enumerate(sm_notes[5]):
        ppm = len(measure)
        for i, step in enumerate(measure):
          beat = measure_num * 4.0 + 4.0 * (float(i) / ppm)
          measure_spec = (measure_num, ppm, i)
          time = btc.beat_to_time(beat)
          measure_beat_time_steps.append((measure_spec, beat, time, step))

      notes = {
        'type': sm_notes[0],
        'stepper': sm_notes[1],
        'difficulty': sm_notes[2],
        'difficulty_fine': sm_notes[3],
        'steps': measure_beat_time_steps,
      }
      out_json['charts'].append(notes)

    # Output JSON
    out_json_fp = os.path.join(pack_out_dir, '{}.raw.json'.format(identifier))
    with open(out_json_fp, 'w') as out_f:
      try:
        out_f.write(json.dumps(out_json))
      except UnicodeDecodeError:
        logger.error('Unicode error in {}'.format(sm_fp))
        continue

    logger.info('Parsed {} - {}: {} charts'.format(pack_ezname, song_ezname, len(out_json['charts'])))

  if delete_pack_dir:
    shutil.rmtree(pack_dir)

  return True

_HASH_READ_CHUNK_SIZE = 65536
_ITG_1_SHA256 = 'ebae4e6cc9f97bed358c705c9cd0813dd02048e510e7120a2d75d553cd1c255c'
_ITG_2_SHA256 = '6f1cc7ff1433d7bd138a372657462a9d687e7dc2612acc8cde4eb659baade077'
_FRAX_T3_SM5_SHA256 = 'ef5643fea9acdcf4adbc495cca01dab042be5648faa96221983ab27c7a4942f4'
_FRAX_AA_SM5_SHA256 = '2d3ce3eaa4d343b9d6a122d22ed48418b65f66cdbf332162336f1a4659b80a95'
_FRAX_BB_SM5_SHA256 = 'c273cc352145eb6466432a13e8b5ee6f581ad9f66d9f2936a20112282964e74f'
_FRAX_T3_ITG_SHA256 = 'ecb92115bc3af3e2c12481c5f018d1a27576e6b6e42181ccc7ed47cca197abea'
_FRAX_AA_ITG_SHA256 = '34ceda918c0531b99293f8231fe2e8beb1aa847417606ff56484858e39373ded'
_FRAX_BB_ITG_SHA256 = '0c0a0901d482da3beeaeb5dd4702d05707035ffaa1039ca1797f2f86957f6c0a'
_KNOWN_SHA256 = [
  _ITG_1_SHA256, _ITG_2_SHA256,
  _FRAX_T3_SM5_SHA256, _FRAX_AA_SM5_SHA256, _FRAX_BB_SM5_SHA256,
  _FRAX_T3_ITG_SHA256, _FRAX_AA_ITG_SHA256, _FRAX_BB_ITG_SHA256
]

_ITG_TAG = 'ITG'
_FRAX_TAG = 'Fraxtil'

if __name__ == '__main__':
  import argparse
  import hashlib
  import os
  import sys
  import zipfile

  parser = argparse.ArgumentParser()
  parser.add_argument('zip_fps', type=str, nargs='+', help='List of zip files to analyze')
  parser.add_argument('out_dir', type=str, help='Output directory for dataset')

  args = parser.parse_args()

  # Create output directory
  if not os.path.isdir(args.out_dir):
    try:
      os.mkdir(args.out_dir)
    except:
      logger.critical('Could not create directory {}'.format(args.out_dir))
      sys.exit(1)

  # Create tmp directory
  tmp_dir = os.path.join(args.out_dir, 'tmp')
  try:
    if os.path.isdir(tmp_dir):
      shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
  except:
    logger.critical('Could not create temporary file directory {}'.format(tmp_dir))
    sys.exit(1)

  # Iterate zips
  for zip_fp in args.zip_fps:
    if not os.path.exists(zip_fp):
      logger.error('Zip file {} does not exist'.format(zip_fp))
      continue

    sha256 = hashlib.sha256()
    with open(zip_fp, 'rb') as f:
      while True:
        data = f.read(_HASH_READ_CHUNK_SIZE)
        if not data:
          break
        sha256.update(data)
    checksum = sha256.hexdigest()

    if checksum in _KNOWN_SHA256:
      with zipfile.ZipFile(zip_fp, 'r') as f:
        f.extractall(tmp_dir)

    if checksum == _ITG_1_SHA256:
      pack_root_name = 'In The Groove'
      tag = _ITG_TAG
      itg_offset = True
    elif checksum == _ITG_2_SHA256:
      pack_root_name = 'In The Groove 2'
      tag = _ITG_TAG
      itg_offset = True
    elif checksum == _FRAX_T3_SM5_SHA256:
      pack_root_name = 'Tsunamix III'
      tag = _FRAX_TAG
      itg_offset = False
    elif checksum == _FRAX_AA_SM5_SHA256:
      pack_root_name = 'Fraxtil\'s Arrow Arrangements'
      tag = _FRAX_TAG
      itg_offset = False
    elif checksum == _FRAX_BB_SM5_SHA256:
      pack_root_name = 'Fraxtil\'s Beast Beats'
      tag = _FRAX_TAG
      itg_offset = False
    elif checksum == _FRAX_T3_ITG_SHA256:
      pack_root_name = 'Tsunamix III'
      tag = _FRAX_TAG
      itg_offset = True
    elif checksum == _FRAX_AA_ITG_SHA256:
      pack_root_name = 'Fraxtil\'s Arrow Arrangements'
      tag = _FRAX_TAG
      itg_offset = True
    elif checksum == _FRAX_BB_ITG_SHA256:
      pack_root_name = 'Fraxtil\'s Beast Beats'
      tag = _FRAX_TAG
      itg_offset = True
    else:
      logger.error('Unrecognized zip file {}'.format(zip_fp))
      continue

    logger.info('Found {} zip'.format(pack_root_name))
    parse_pack(
      os.path.join(tmp_dir, pack_root_name),
      args.out_dir,
      tag,
      itg_offset=itg_offset,
      delete_pack_dir=True)

  # Delete tmp directory
  try:
    shutil.rmtree(tmp_dir)
  except:
    logger.error('Could not delete temporary file directory {}'.format(tmp_dir))
