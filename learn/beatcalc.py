import numpy as np

_EPSILON = 1e-6
class BeatCalc(object):
    # for simplicity, we will represent a "stop" as an impossibly sharp tempo change
    def __init__(self, offset, beat_bpm, beat_stop):
        # ensure all beat markers are strictly increasing
        assert beat_bpm[0][0] == 0.0
        beat_last = -1.0
        bpms = []
        for beat, bpm in beat_bpm:
            assert beat >= beat_last
            if beat == beat_last:
                bpms[-1] = (beat, bpm)
            else:
                bpms.append((beat, bpm))
            beat_last = beat

        # aggregate repeat stops
        stops = {}
        for beat, stop in beat_stop:
            assert beat > 0.0
            if beat in stops:
                stops[beat] += stop
            stops[beat] = stop
        beat_stop = filter(lambda x: x[1] != 0.0, sorted(stops.items(), key=lambda x: x[0]))

        self.offset = offset
        self.bpms = beat_bpm
        self.stops = beat_stop

        beat_bps = [(beat, bpm / 60.0) for beat, bpm in beat_bpm]

        # insert line segments for stops
        for beat, stop in beat_stop:
            seg_idx = np.searchsorted(np.array([x[0] for x in beat_bps]), beat, side='right')
            _, bps = beat_bps[seg_idx - 1]

            beat_bps.insert(seg_idx, (beat + _EPSILON, bps))
            beat_bps.insert(seg_idx, (beat, _EPSILON / stop))

        # create line segments for tempo changes
        time_cum = -offset
        beat_last, bps_last = beat_bps[0]
        times = [-offset]
        for beat, bps in beat_bps[1:]:
            dbeat = beat - beat_last
            dtime = dbeat / bps_last
            time_cum += dtime
            times.append(time_cum)
            beat_last = beat
            bps_last = bps

        self.segment_time = np.array(times)
        self.segment_beat = np.array([beat for beat, _ in beat_bps])
        self.segment_bps = np.array([bps for _, bps in beat_bps])
        self.segment_spb = 1.0 / self.segment_bps

    def beat_to_time(self, beat):
        assert beat >= 0.0
        seg_idx = np.searchsorted(self.segment_beat, beat, side='right') - 1
        beat_left = self.segment_beat[seg_idx]
        time_left = self.segment_time[seg_idx]
        spb = self.segment_spb[seg_idx]
        return time_left + ((beat - beat_left) * spb)

    def time_to_beat(self, time):
        assert time >= 0.0
        seg_idx = np.searchsorted(self.segment_time, time, side='right') - 1
        time_left = self.segment_time[seg_idx]
        beat_left = self.segment_beat[seg_idx]
        bps = self.segment_bps[seg_idx]
        return beat_left + ((time - time_left) * bps)

if __name__ == '__main__':
    bc = BeatCalc(0.05, [(0.0, 120.0), (32.0, 60.0), (64.0, 120.0)], [(16.0, 5.0)])
    print bc.beat_to_time(0.0)
    print bc.beat_to_time(1.0)
    print bc.beat_to_time(8.0)
    print bc.beat_to_time(16.0)
    print bc.beat_to_time(32.0)
    print '-' * 80
    print bc.time_to_beat(0.0)
    print bc.time_to_beat(1.0)
