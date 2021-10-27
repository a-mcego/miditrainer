import hashlib
import glob
import pathlib

digests = {}

midis = glob.glob(f"Q:\\midi_lakh\\*\\*.mid")
for midi in midis:
    hashnum = midi.split("\\")[-1][:-4]
    print(hashnum, end="    \r")
    digests[hashnum] = midi

lakhmidis_n = len(midis)
dupes = 0
#midis = glob.glob(f"Q:\\midi\\*\\*.mid")
midis = list(pathlib.Path('q:\\gamemidi').rglob('*.mid'))
for midi in midis:
    print(f"{midi}", end="     \r")
    with open(midi, "rb") as mfile:
        d = hashlib.md5(mfile.read()).hexdigest()
        if d in digests:
            dupes += 1
            print(midi)
            
midis_n = len(midis)

print(f"{midis_n} in the old dataset")
print(f"{lakhmidis_n} in the new lakh dataset")
print(f"{dupes} dupes found.")