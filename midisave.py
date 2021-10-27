import mido
from mido.midifiles.midifiles import DEFAULT_TEMPO
from mido.midifiles.tracks import MidiTrack, merge_tracks, fix_end_of_track
import numpy as np

version = "v4"

DELAY_PRECISION=0.001 #how many seconds the delay of 1 has

def convert_delay(x):
    if x > 3008:
        x = 3008
    if x<128:
        return x
    if x<256:
        return (x-128)//2+128
    if x<512:
        return (x-256)//4+192
    if x<1024:
        return (x-512)//8+256
    if x<1536:
        return (x-1024)//16+320
    if x<2048:
        return (x-1536)//32+352
    return (x-2048)//64+368

#1ms resolution from 0ms to 127ms: 0 -> 127
#2ms resolution from 128ms to 254ms: 128 -> 191
#4ms resolution from 256ms to 508ms: 192 -> 255
#8ms resolution from 512ms to 1016ms 256 -> 319
#16ms resolution from 1024ms to 1520ms 320 -> 351
#32ms resolution from 1536ms to 2016ms 352 -> 367
#64ms resolution from 2048ms to 3008ms 368 -> 383
def unconvert_delay(x):
    ret=0
    if x<128:
        ret=x
    elif x<192:
        ret=(x-128)*2+128
    elif x<256:
        ret=(x-192)*4+256
    elif x<320:
        ret=(x-256)*8+512
    elif x<352:
        ret=(x-320)*16+1024
    elif x<368:
        ret=(x-352)*32+1536
    else:
        ret=(x-368)*64+2048
    return ret

#these are the controller #s supported by General MIDI
id_to_controller = [1, 7, 10, 11, 64, 100, 101, 121, 123]
N_CONTROLLERS = len(id_to_controller)
controller_to_id = {}
for c in range(N_CONTROLLERS):
    controller_to_id[id_to_controller[c]] = c


#139 comes from 128 notes, 10 controllers (incl. pitch bend) and 1 delay


#NOTE:       note      , velocity, channel, program, length
#CONTROLLER: controller, value   , channel, 0      , 0
#DELAY:      delay     , 0       , 0      , 0      , length
#
#vocab_size: 128+10+1  , 128+128 , 16     , 128    , 384#
#            140       , 256     , 16     , 128    , 384
#total = 923


"""TOKEN_NCD     = 0
TOKEN_VELVAL  = 139
TOKEN_CHANNEL = 139+256
TOKEN_PROGRAM = 139+256+16
TOKEN_LENGTH  = 139+256+16+128
N_TOKEN_TOTAL = 139+256+16+128+384"""

TOKEN_CHANNEL_PROGRAM  = 0                                                  #   0-2047
TOKEN_DELAY            = 16*128                                             #2048-2431
TOKEN_NOTE_ON          = 16*128+384                                         #2432-2559    
TOKEN_NOTE_ON_DRUMS    = 16*128+384+128                                     #2560-2687
TOKEN_NOTE_OFF         = 16*128+384+128+128                                 #2688-2815
TOKEN_VELOCITY         = 16*128+384+128+128+128                             #2816-2943
TOKEN_CONTROLLER_TYPE  = 16*128+384+128+128+128+128                         #2944-2952
TOKEN_CONTROLLER_VALUE = 16*128+384+128+128+128+128+N_CONTROLLERS           #2953-3080
TOKEN_PITCHBEND        = 16*128+384+128+128+128+128+N_CONTROLLERS+128       #3081-3209
N_TOKEN_TOTAL          = 16*128+384+128+128+128+128+N_CONTROLLERS+128+128   #^ TOKEN ID RANGES


token_to_ms = np.array([unconvert_delay(x)*DELAY_PRECISION for x in range(TOKEN_NOTE_ON-TOKEN_DELAY)], dtype=np.float32)
token_to_ms = np.concatenate([np.zeros(TOKEN_DELAY, dtype=np.float32), token_to_ms, np.zeros(N_TOKEN_TOTAL-TOKEN_NOTE_ON, dtype=np.float32)], axis=-1)
assert token_to_ms.shape[0] == N_TOKEN_TOTAL, "token_to_ms is the wrong shape in midisave.py"


def load_midi(filename, clip):
    try:
        mid = mido.MidiFile(filename, clip=clip)
    except Exception as err:
        print(f"OPEN {filename}: {type(err)} {err}")
        return None
        
    total_delay=0.0
    
    current_program = [0]*16
    
    #print(f"Good file {filename}")
    
    
    tempo = DEFAULT_TEMPO
    clocks_per_click = mid.ticks_per_beat
    current_channel = None
    
    try:
        out = []
        for msg in merge_tracks(mid.tracks):
            m = msg.dict()
            
            if m['type'] == 'set_tempo':
                tempo = m['tempo']
            
            total_delay += mido.tick2second(m['time'],clocks_per_click,tempo)
            
            if m['type'] in ['text','time_signature','track_name','midi_port','key_signature','copyright','instrument_name','channel_prefix','end_of_track', 'marker', 'sysex', 'smpte_offset', 'lyrics', 'cue_marker', 'sequencer_specific', 'aftertouch', 'polytouch', 'sequence_number', 'set_tempo', 'stop']:
                continue
            
            if m['type'] == 'control_change' and m['control'] not in id_to_controller:
                continue
            
            stime = convert_delay(round(total_delay*(1.0/DELAY_PRECISION)))
            if stime != 0 and m['time'] != 0:
                out.extend([TOKEN_DELAY+stime])
                total_delay = max(0.0,total_delay-unconvert_delay(stime)*DELAY_PRECISION)
            
            if m['type'] == 'note_on' or m['type'] == 'note_off':
                if current_channel != m['channel']:
                    current_channel = m['channel']
                    out.extend([TOKEN_CHANNEL_PROGRAM+m['channel']+current_program[m['channel']]*16])
                
                if m['velocity'] == 0 or m['type'] == 'note_off':
                    if current_channel != 9:
                        out.extend([TOKEN_NOTE_OFF+m['note']])
                else:
                    if current_channel == 9:
                        out.extend([TOKEN_NOTE_ON_DRUMS+m['note'], TOKEN_VELOCITY+m['velocity']])
                    else:
                        out.extend([TOKEN_NOTE_ON+m['note'], TOKEN_VELOCITY+m['velocity']])
            
            elif m['type'] == 'control_change':
                if m['control'] in id_to_controller:
                    current_channel = m['channel']
                    out.extend([TOKEN_CHANNEL_PROGRAM+m['channel']+current_program[m['channel']]*16])
                    out.extend([TOKEN_CONTROLLER_TYPE+controller_to_id[m['control']], TOKEN_CONTROLLER_VALUE+m['value']])
            elif m['type'] == 'pitchwheel':
                if current_channel != m['channel']:
                    current_channel = m['channel']
                    out.extend([TOKEN_CHANNEL_PROGRAM+m['channel']+current_program[m['channel']]*16])

                out.extend([TOKEN_PITCHBEND+(m['pitch']+8192)//128])
            elif m['type'] == 'program_change':
                current_channel = m['channel']
                current_program[current_channel] = m['program']
                out.extend([TOKEN_CHANNEL_PROGRAM+m['channel']+m['program']*16])

    except Exception as err:
        print(f"MERGE {filename}: {type(err)} {err}")
        return None

    out.extend([TOKEN_DELAY+convert_delay(5000)])    
    return out

def save_midi(data, filename):
    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    current_channel = 0
    current_delay = 0.0
    current_controller = None
    current_noteon = -1
    current_program = [0]*16

    tempo = DEFAULT_TEMPO
    clocks_per_click = mid.ticks_per_beat
    
    total_length = 0.0

    def get_delay():
        nonlocal current_delay
        tick = int(mido.second2tick(current_delay,clocks_per_click,tempo))
        current_delay -= mido.tick2second(tick,clocks_per_click,tempo)
        if current_delay < 0.0:
            current_delay = 0.0
        return tick
        
    
    for d in data:
        if d>=TOKEN_PITCHBEND:
            track.append(mido.Message('pitchwheel', channel=current_channel, pitch=(d-TOKEN_PITCHBEND)*128-8192, time=get_delay()))
        elif d>=TOKEN_CONTROLLER_VALUE:
            if current_controller is not None:
                track.append(mido.Message('control_change', channel=current_channel, control=id_to_controller[current_controller], value=d-TOKEN_CONTROLLER_VALUE, time=get_delay()))
        elif d>=TOKEN_CONTROLLER_TYPE:
            current_controller = d-TOKEN_CONTROLLER_TYPE
        elif d>=TOKEN_VELOCITY:
            if current_noteon != -1:
                track.append(mido.Message('note_on', channel=current_channel, time=get_delay(), note=current_noteon, velocity=d-TOKEN_VELOCITY))
                current_noteon = -1
        elif d>=TOKEN_NOTE_OFF:
            track.append(mido.Message('note_off', channel=current_channel, time=get_delay(), note=d-TOKEN_NOTE_OFF, velocity=0))
        elif d>=TOKEN_NOTE_ON_DRUMS:
            current_noteon = d-TOKEN_NOTE_ON_DRUMS
        elif d>=TOKEN_NOTE_ON:
            current_noteon = d-TOKEN_NOTE_ON
        elif d>=TOKEN_DELAY:
            current_delay += unconvert_delay(d-TOKEN_DELAY)*DELAY_PRECISION
            total_length += unconvert_delay(d-TOKEN_DELAY)*DELAY_PRECISION
        elif d>=TOKEN_CHANNEL_PROGRAM:
            data = d-TOKEN_CHANNEL_PROGRAM
            current_channel = data%16
            if current_program[current_channel] != data//16:
                track.append(mido.Message('program_change', channel=current_channel, program=data//16, time=get_delay()))
                current_program[current_channel] = data//16

    print(len(track), total_length, end=" ")
    mid.save(filename)
