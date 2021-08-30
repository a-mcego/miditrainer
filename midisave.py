import mido
from mido.midifiles.midifiles import DEFAULT_TEMPO
from mido.midifiles.tracks import MidiTrack, merge_tracks, fix_end_of_track

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

TOKEN_CHANNEL          = 0
TOKEN_DELAY            = 16
TOKEN_NOTE_ON          = 16+384
TOKEN_NOTE_OFF         = 16+384+128
TOKEN_VELOCITY         = 16+384+128+128
TOKEN_CONTROLLER_TYPE  = 16+384+128+128+128
TOKEN_CONTROLLER_VALUE = 16+384+128+128+128+128
TOKEN_PITCHBEND        = 16+384+128+128+128+128+128
TOKEN_PROGRAM          = 16+384+128+128+128+128+128+128
N_TOKEN_TOTAL          = 16+384+128+128+128+128+128+128+128

def load_midi(filename, clip):
    try:
        mid = mido.MidiFile(filename, clip=clip)
    except Exception as err:
        #print(f"OPEN {filename}: {type(err)} {err}")
        return None
        
    total_delay=0.0
    
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
            
            stime = convert_delay(round(total_delay*1000))
            if stime != 0 and m['time'] != 0:
                out.extend([TOKEN_DELAY+stime])
                total_delay = max(0.0,total_delay-unconvert_delay(stime)*0.001)
            
            if m['type'] == 'note_on' or m['type'] == 'note_off':
                if current_channel != m['channel']:
                    current_channel = m['channel']
                    out.extend([TOKEN_CHANNEL+m['channel']])
                
                if m['velocity'] == 0 or m['type'] == 'note_off':
                    out.extend([TOKEN_NOTE_OFF+m['note']])
                else:
                    out.extend([TOKEN_NOTE_ON+m['note'], TOKEN_VELOCITY+m['velocity']])
            
            elif m['type'] == 'control_change':
                current_channel = m['channel']
                out.extend([TOKEN_CHANNEL+m['channel']])
                out.extend([TOKEN_CONTROLLER_TYPE+m['control'], TOKEN_CONTROLLER_VALUE+m['value']])
            elif m['type'] == 'pitchwheel':
                if current_channel != m['channel']:
                    current_channel = m['channel']
                    out.extend([TOKEN_CHANNEL+m['channel']])

                out.extend([TOKEN_PITCHBEND+(m['pitch']+8192)//128])
            elif m['type'] == 'program_change':
                current_channel = m['channel']
                out.extend([TOKEN_CHANNEL+m['channel']])
                out.extend([TOKEN_PROGRAM+m['program']])

    except Exception as err:
        #print(f"MERGE {filename}: {type(err)} {err}")
        return None

    out.extend([TOKEN_DELAY+convert_delay(5000)])    
    return out

def save_midi(data, filename):
    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    current_channel = 0
    current_delay = 0.0
    current_controller = 0
    current_noteon = -1

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
        if d>=TOKEN_PROGRAM:
            track.append(mido.Message('program_change', channel=current_channel, program=d-TOKEN_PROGRAM, time=get_delay()))
        elif d>=TOKEN_PITCHBEND:
            track.append(mido.Message('pitchwheel', channel=current_channel, pitch=(d-TOKEN_PITCHBEND)*128-8192, time=get_delay()))
        elif d>=TOKEN_CONTROLLER_VALUE:
            track.append(mido.Message('control_change', channel=current_channel, control=current_controller, value=d-TOKEN_CONTROLLER_VALUE, time=get_delay()))
        elif d>=TOKEN_CONTROLLER_TYPE:
            current_controller = d-TOKEN_CONTROLLER_TYPE
        elif d>=TOKEN_VELOCITY:
            if current_noteon != -1:
                track.append(mido.Message('note_on', channel=current_channel, time=get_delay(), note=current_noteon, velocity=d-TOKEN_VELOCITY))
                current_noteon = -1
        elif d>=TOKEN_NOTE_OFF:
            track.append(mido.Message('note_off', channel=current_channel, time=get_delay(), note=d-TOKEN_NOTE_OFF, velocity=0))
        elif d>=TOKEN_NOTE_ON:
            current_noteon = d-TOKEN_NOTE_ON
        elif d>=TOKEN_DELAY:
            current_delay += unconvert_delay(d-TOKEN_DELAY)*0.001
            total_length += unconvert_delay(d-TOKEN_DELAY)*0.001
        elif d>=TOKEN_CHANNEL:
            current_channel = d-TOKEN_CHANNEL

    print(len(track), total_length, end=" ")
    mid.save(filename)
