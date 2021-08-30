import numpy as np
import scipy
import scipy.special
import time
import mido
import pathlib
from mido.midifiles.tracks import MidiTrack, merge_tracks, fix_end_of_track
from mido.midifiles.midifiles import DEFAULT_TEMPO
import sys
import glob

import midisave

#midis = glob.glob(f"Q:\\midi\\*\\*.mid")
#savefilename = "training_tokens.npy"
#clip = False

#midis = glob.glob("C:\\Users\\liimaeve\\Dropbox\\koodaus2\\GitHub\\miditrainer\\validation_midis\\*.mid")
#savefilename = "validation_tokens_v2.npy"
#clip = True

midis = list(pathlib.Path('q:\\gamemidi').rglob('*.mid'))
print(f"{len(midis)} midis found.")
savefilename = "gamemidi_tokens_test.npy"
clip = True

#print(midis)

n_good = 0
n_total = 0

BLOCKS = 400000000

total_out = np.zeros(BLOCKS, dtype=np.int16)
total_place = 0

n_totals = [0]*midisave.N_TOKEN_TOTAL
for filename in midis:
    n_total += 1
    
    out = midisave.load_midi(filename, clip)
    if out is None:
        continue

    for t in out:
        n_totals[t] += 1
        
    out = np.array(out, dtype=np.int16)
    n_good += 1
    
    p = np.array(n_totals)/sum(n_totals)
    q = np.full(p.shape, 1.0/len(n_totals))
    
    kl_div = np.sum(scipy.special.xlogy(p,p) - scipy.special.xlogy(p,q))
    
    while total_out.shape[0] < sum(n_totals):
        total_out = np.concatenate([total_out,np.zeros([BLOCKS], dtype=np.int16)],axis=0)
        
    total_out[total_place:(total_place+out.shape[0])] = out
    total_place += out.shape[0]
    
    print(f"{n_good}/{n_total} ... with sum {sum(n_totals)}, total~={round(sum(n_totals)/n_total*len(midis)/1000000.0)}M KL={kl_div}", end = "     \r", flush=True)
    
print()
print(f"{n_good}/{n_total} ... with sum {sum(n_totals)}, total~={round(sum(n_totals)/n_total*len(midis)/1000000.0)}M KL={kl_div}")

total_out = total_out[:sum(n_totals)]

np.save(savefilename, total_out)

#print(len(out))
