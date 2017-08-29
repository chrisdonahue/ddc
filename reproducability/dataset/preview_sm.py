import json
import sys

_TEMPL = """\
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music_fp};
#OFFSET:0.0;
#BPMS:0.0={bpm};
#STOPS:;
{charts}\
"""

_CHART_TEMPL = """\
#NOTES:
    {ctype}:
    {cversion}:
    {ccoarse}:
    {cfine}:
    0.500,0.500,0.500,0.500,0.500:
{measures};\
"""

def meta_to_sm(meta):
    subdiv = 64
    dt = 512.0 / 44100.0
    # seconds per minute * timesteps per second * measures per subdivision * beats per measure
    bpm = 60 * (1.0 / dt) * (1.0 / float(subdiv)) * 4.0

    charts = []
    for chart in meta['charts']:
        ctype = chart['type']
        cversion = 'Stepnet'
        ccoarse = chart['difficulty_coarse']
        cfine = chart['difficulty_fine']
        cnotes = chart['notes']
        
        measures = []
        timestep_to_code = {int(round(t / dt)) : code for _, t, code in cnotes}
        max_s = cnotes[-1][1] + 15.0
        max_timestep = int(round(max_s / dt))
        if max_timestep % subdiv != 0:
            max_timestep += subdiv - (max_timestep % subdiv)

        null_code = '0' * len(cnotes[0][2])
        timesteps = [timestep_to_code.get(i, null_code) for i in xrange(max_timestep)]
        measures = [timesteps[i:i+subdiv] for i in xrange(0, max_timestep, subdiv)]
        measures_txt = '\n,\n'.join(['\n'.join(measure) for measure in measures])

        chart_txt = _CHART_TEMPL.format(
            ctype=ctype,
            cversion=cversion,
            ccoarse=ccoarse,
            cfine=cfine,
            cgroove='',
            measures=measures_txt)

        charts.append(chart_txt)

    return _TEMPL.format(
        title=meta['title'],
        artist=meta['artist'],
        music_fp=meta['music_fp'],
        bpm=bpm,
        charts='\n'.join(charts))

if __name__ == '__main__':
    json_fp, sm_fp = sys.argv[1:3]
    with open(json_fp, 'r') as f:
        meta = json.loads(f.read())

    sm_txt = meta_to_sm(meta)

    with open(sm_fp, 'w') as f:
        f.write(sm_txt)