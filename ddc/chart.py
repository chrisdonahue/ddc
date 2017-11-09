import os

from ddc import BeatTimeCalc


class Song(object):
  def __init__(self, file_dir, attrs):
    self.ezpack = attrs['pack_name']
    self.eztitle = attrs['song_name']

    self.artist = attrs['artist']
    self.title = attrs['title']

    self.btcalc = BeatTimeCalc(attrs['offset'], attrs['bpms'], attrs['stops'])

    self.audio_fp = os.path.join(file_dir, attrs['music_fp'])

    self.charts_raw = attrs['charts']
    self.charts = None


  def get_audio_fp(self):
    return self.audio_fp


  def get_placement_charts(self):
    if self.charts is None:
      self.charts = []
      for chart_attrs in self.charts_raw:
        self.charts.append(PlacementChart(self, chart_attrs))
    return self.charts


class Chart(object):
  def __init__(self, song, chart_attrs):
    self.song = song

    self.difficulty = chart_attrs['difficulty_coarse']
    self.difficulty_fine = chart_attrs['difficulty_fine']
    self.type = chart_attrs['type']
    self.stepper = chart_attrs['desc_or_author']

    self.steps = chart_attrs['notes']

  def get_difficulty():
    return self.difficulty


class PlacementChart(Chart):
  def __init__(self, song, chart_attrs):
    super(PlacementChart, self).__init__(song, chart_attrs)


  def get_audio_fp(self):
    return self.song.get_audio_fp()


  def get_feats(self,
      difficulty=True):
    feats = {}
    if difficulty:
      feats['difficulty'] = self.difficulty

    return feats


  def get_step_frames(self, rate):
    return [int(round(t * rate)) for _, _, t, _ in self.steps]
