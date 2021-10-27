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
#savefilename = f"C:\\datasets\\midi\\training_tokens_{midisave.version}.npy"
#clip = False

midis = glob.glob(f"Q:\\midi_lakh\\*\\*.mid")
savefilename = f"C:\\datasets\\midi\\lakh_training_tokens_{midisave.version}.npy"
clip = True

#midis = glob.glob("C:\\datasets\\midi\\validation_midis\\*.mid")
#savefilename = f"C:\\datasets\\midi\\validation_tokens_{midisave.version}.npy"
#clip = True

#midis = list(pathlib.Path('q:\\gamemidi').rglob('*.mid'))
#savefilename = f"C:\\datasets\\midi\\gamemidi_tokens_{midisave.version}.npy"
#clip = True

print(f"{len(midis)} midis found.")
#print(midis)

def test_midis(midis):
    goods = 0
    totals = 0
    
    #midis2 = ['Q:\\midi_lakh\\0\\007e052394dee52b75d6a5cf1ed0d561.mid']
    
    for filename in midis:
        errors = midisave.test_midi(filename)
        if errors is None:
            pass
        elif errors == 0:
            goods += 1
            totals += 1
        else:
            totals += 1
            print(f"{filename}: {errors} errors!")
        
        print(f"{goods} / {totals}     ",end="\r")

        if totals == 400:
            break

    print()
    exit()

#test_midis(midis)

n_good = 0
n_total = 0

BLOCKS = 400000000

total_out = np.zeros([BLOCKS,5], dtype=np.int16)
total_place = 0

n_tokens_total = 0

#n_totals = [0]*midisave.N_TOKEN_TOTAL
for filename in midis:
    n_total += 1
    
    out = midisave.load_midi(filename, clip)
    if out is None:
        continue

    n_tokens_total += len(out)

    #for t in out:
    #    n_totals[t] += 1
        
    out = np.array(out, dtype=np.int16)
    n_good += 1
    
    #p = np.array(n_totals)/sum(n_totals)
    #q = np.full(p.shape, 1.0/len(n_totals))
    
    #kl_div = np.sum(scipy.special.xlogy(p,p) - scipy.special.xlogy(p,q))
    
    while total_out.shape[0] < n_tokens_total:
        total_out = np.concatenate([total_out,np.zeros([BLOCKS,5], dtype=np.int16)],axis=0)
        
    total_out[total_place:(total_place+out.shape[0])] = out
    total_place += out.shape[0]
    
    #print(f"{n_good}/{n_total} ... with sum {sum(n_totals)}, total~={round(sum(n_totals)/n_total*len(midis)/1000000.0)}M KL={kl_div}", end = "     \r", flush=True)
    print(f"{n_good}/{n_total}, total~={round(n_tokens_total/n_total*len(midis)/1000000.0)}M", end = "     \r", flush=True)
    
print()
#print(f"{n_good}/{n_total} ... with sum {sum(n_totals)}, total~={round(sum(n_totals)/n_total*len(midis)/1000000.0)}M KL={kl_div}")
print(f"{n_good}/{n_total}, total~={round(n_tokens_total/n_total*len(midis)/1000000.0)}M", end = "     \r", flush=True)

total_out = total_out[:n_tokens_total]

np.save(savefilename, total_out)

#print(len(out))
