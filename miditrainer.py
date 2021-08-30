#this does the actual training

import random
import sys
#sys.tracebacklimit = 0
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
np.set_printoptions(precision=4,suppress=True)
import tensorflow as tf
import time
import enum
import matplotlib.pyplot as plt
import datahandler
import midisave

from counting import Counting
from transformer import Transformer
import posenc



def augment(d1,d2,tokens, N_MAX):
    r = np.arange(len(tokens))
    np.random.shuffle(r)
    new_tokens = tokens[r]
    replacer = np.arange(N_MAX)
    replacer[tokens] = new_tokens
    for f in range(d1.shape[0]):
        d1[f] = replacer[d1[f]]
    for f in range(d2.shape[0]):
        d2[f] = replacer[d2[f]]

def augment_channels(d1, d2, N_MAX):
    tokens = np.array([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15])
    augment(d1,d2,tokens, N_MAX)

if len(sys.argv) != 2:
    print("Give log save name as argument.")
    exit(0)
if not os.path.exists("outs"):
    os.mkdir("outs")
    
def set_gpu_settings():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def jsonload(filename):
    return json.load(open(filename,"r"))
    
def jsonsave(obj, filename):
    json.dump(obj, open(filename, "w"))
    
set_gpu_settings()

prm = {} #parameters
prm['PRINT_STEPS'] = 64
prm['VALIDATION_STEPS'] = prm['PRINT_STEPS']*8
prm['GEN_STEPS'] = prm['PRINT_STEPS']*16
prm['SAVE_STEPS'] = prm['PRINT_STEPS']*32
prm['MODEL_SIZE'] = 768
prm['XF_LAYERS'] = 6
prm['XF_HEADS'] = 6
prm['LEARNING_RATE'] = 0.001
prm['LEARNING_RATE_MIN'] = 0.001
prm['ADAM_EPSILON'] = 1e-4
prm['BATCH_SIZE'] = 16
prm['WARMUP_STEPS'] = 0
prm['N_TIMESTEPS'] = 512
prm['VOCAB_SIZE'] = midisave.N_TOKEN_TOTAL
prm['GEN_LENGTH'] = prm['N_TIMESTEPS']*3
prm['TOP_K'] = 10
prm['MODEL_SAVE_DIR'] = "Q:\\midi_model_saves\\" + sys.argv[1]

for key in prm:
    print(f"{key}={prm[key]} ",end="")
print()

if not os.path.exists(prm['MODEL_SAVE_DIR']):
    os.mkdir(prm['MODEL_SAVE_DIR'])

steps = 1

def remove_braces(stuff):
    return stuff.replace('[','').replace(']','')

#custom to string.
def cts(stuff):
    ret = str([x.numpy() for x in stuff])
    return remove_braces(ret)
def cts_numpy(stuff):
    ret = str([x for x in stuff])
    return remove_braces(ret)
def cts2(stuff):
    ret = str(stuff)
    return remove_braces(ret)


#data = datahandler.NumpyFileShuffle(prm['N_TIMESTEPS'], filename_t="C:\\datasets\\midi\\training_tokens_v1.npy", filename_v="C:\\datasets\\midi\\validation_tokens_v1.npy", vocab_size=prm['VOCAB_SIZE'])

data = datahandler.NumpyFileShuffle(prm['N_TIMESTEPS'], filename_t="C:\\datasets\\midi\\gamemidi_tokens.npy", filename_v="C:\\datasets\\midi\\validation_tokens_v1.npy", vocab_size=prm['VOCAB_SIZE'])

#unique, counts = np.unique(data.tokens_t, return_counts=True)
#print(list(zip(unique,counts)))
#exit(0)


class TaskSolver(tf.keras.Model):
    def __init__(self):
        super(TaskSolver,self).__init__()

        self.embedding = tf.keras.layers.Embedding(prm['VOCAB_SIZE'], prm['MODEL_SIZE'])
        self.posenc = tf.convert_to_tensor(posenc.get_standard_posenc(prm['N_TIMESTEPS'], prm['MODEL_SIZE']))*0.05

        self.xformer = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=True, use_counter=False, normalize_columns=False)
        
        self.logit_outputs = tf.keras.layers.Dense(prm['VOCAB_SIZE'])
        
    def call(self, n_input):
        n_output = n_input
        n_output = self.embedding(n_output) + self.posenc[:n_input.shape[-1]]
        n_output = self.xformer(n_output)
        n_output = self.logit_outputs(n_output)
        return n_output
        
tasksolver = TaskSolver()

def lr_scheduler():
    global steps
    
    warmup = prm['WARMUP_STEPS']
    if steps < warmup:
        #steps+1 because otherwise it leads to a learning rate of *zero*
        #which messes up everything!
        return prm['LEARNING_RATE']*float(steps+1)/float(warmup)
    minimum = prm['LEARNING_RATE_MIN']
    calculated = prm['LEARNING_RATE']*(2.0**(-(steps-warmup)/400.0))
    return max(minimum,calculated)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,amsgrad=True,epsilon=prm['ADAM_EPSILON'])

start_time = time.time()

@tf.function
def do_step(n_input, target, training):
    losses = []
    accuracies_c = []
    
    if training:
        with tf.GradientTape() as tape:
            n_output = tasksolver(n_input)
            loss_c = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
              target, n_output, from_logits=True))
            pred_ids = tf.keras.backend.argmax(n_output,axis=-1)
            losses.append(loss_c)
        gradients = tape.gradient(losses, tasksolver.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, tasksolver.trainable_variables))
    else:
        n_output = tasksolver(n_input)
        loss_c = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
          target, n_output, from_logits=True))
        pred_ids = tf.keras.backend.argmax(n_output,axis=-1)
        losses.append(loss_c)

    correct_chars = tf.reduce_mean(tf.cast(tf.equal(pred_ids,target),dtype=tf.float32),axis=-1)
    accuracy_c = tf.squeeze(tf.reduce_mean(correct_chars))
    accuracies_c.append(accuracy_c)

    return losses, accuracies_c, pred_ids

all_losses = []
all_accs_char = []
    
def write_header(myfile):
    myfile.write(f"------------------------------------------------------------------\n")
    myfile.write(f"NEW MODEL! :D {training_sess_id}\n")
    for key in prm:
        myfile.write(f"{key}={prm[key]} ")
    myfile.write("\n")
    
training_sess_id = random.randrange(1000000000)
print(f"Rnd num: {training_sess_id}")

if os.path.isfile(prm['MODEL_SAVE_DIR']+'/settings.txt'):
    settings = jsonload(prm['MODEL_SAVE_DIR']+'/settings.txt')
    
    checkpoint = tf.train.Checkpoint(opt=optimizer, model=tasksolver)
    status = checkpoint.restore(tf.train.latest_checkpoint(prm['MODEL_SAVE_DIR']+'/weights'))
    
    steps = settings['steps']
    training_sess_id = settings['training_sess_id']

with open("training_log.txt", "a", encoding='utf8') as myfile:
    write_header(myfile)

while True:
    input_data = []
    targets = []
    for _ in range(prm['BATCH_SIZE']):
        n_input, target = data.get_random_t_data(prm['N_TIMESTEPS'])
        
        augment_channels(n_input, target, prm['VOCAB_SIZE'])
        
        input_data.append(n_input)
        targets.append(target)
        
    input_data = tf.stack(input_data,axis=0)
    targets = tf.stack(targets,axis=0)
    
    input_data = tf.cast(input_data, tf.int64)
    targets = tf.cast(targets, tf.int64)
    
    losses, accuracies_char, _ = do_step(input_data, target=targets, training=True)
    
    all_losses.extend(losses)
    all_accs_char.extend(accuracies_char)

    if steps % prm['PRINT_STEPS'] == 0:
        totaltime = time.time()-start_time
        time_per_token = totaltime / float(prm['N_TIMESTEPS']) / float(prm['BATCH_SIZE']) / float(prm['PRINT_STEPS'])
        
        token_per_second = 1.0 / time_per_token
        
        start_time = time.time()

        #HACK
        #for some reason np.set_printoptions doesnt affect scalars
        #so we print a dumdum array instead
        losses = np.array([(tf.add_n(all_losses) / len(all_losses))])
        accs_char = np.array([(tf.add_n(all_accs_char) / len(all_accs_char))])

        with open("training_log.txt", "a", encoding='utf8') as myfile:
            myfile.write(f"{steps} {steps*prm['N_TIMESTEPS']*prm['BATCH_SIZE']} {round(totaltime,4)} s ")
            myfile.write(f"{round(token_per_second*86400.0/1000.0/1000.0/1000.0,4)} Gtok/day ")
            myfile.write(f"l {cts2(losses)} ")
            myfile.write(f"a {cts2(accs_char)} ")
            myfile.write(f"\n")

        print(f"{steps} {steps*prm['N_TIMESTEPS']*prm['BATCH_SIZE']} {round(totaltime,4)} s ", end='')
        print(f"{round(token_per_second*86400.0/1000.0/1000.0/1000.0,4)} Gtok/day ", end='')
        print(f"l {remove_braces(str(losses))} ", end='')
        print(f"a {remove_braces(str(accs_char))} ", end='')
        print(f"lr {lr_scheduler()} ", end='')
        

        if steps % prm['VALIDATION_STEPS'] == 0:
            vdata = data.get_v_data()
            
            vdata_input = vdata[:-1]
            vdata_target = vdata[1:]
            
            vdata_input = vdata_input[:vdata_input.shape[0]-vdata_input.shape[0]%prm['N_TIMESTEPS']]
            vdata_input = np.reshape(vdata_input, [-1, prm['N_TIMESTEPS']])
            vdata_input = tf.cast(vdata_input, tf.int64)

            vdata_target = vdata_target[:vdata_target.shape[0]-vdata_target.shape[0]%prm['N_TIMESTEPS']]
            vdata_target = np.reshape(vdata_target, [-1, prm['N_TIMESTEPS']])
            vdata_target = tf.cast(vdata_target, tf.int64)
            
            v_result = np.zeros(shape=(2,vdata_input.shape[0]), dtype=np.float32)
            
            for start in range(0,vdata_input.shape[0]):
                losses, accuracies, _ = do_step(vdata_input[start:start+1], target=vdata_target[start:start+1], training=False)
                v_result[0,start] = losses[0].numpy()
                v_result[1,start] = accuracies[0].numpy()
                
            v_result = np.mean(v_result, axis=-1)
                
            print(f"vl {cts_numpy([v_result[0]])} ", end="")
            print(f"va {cts_numpy([v_result[1]])} ", end="")
                
            with open("validation_log.txt", "a", encoding='utf8') as myfile:
                myfile.write(f"steps {steps} ")
                myfile.write(f"l {cts_numpy([v_result[0]])} ")
                myfile.write(f"a {cts_numpy([v_result[1]])} ")
                myfile.write(f"\n")

        if steps % prm['GEN_STEPS'] == 0:
            #this one loads a midi 
            #generated = midisave.load_midi("C:\\datasets\\midi\\validation_midis\\spora.mid", clip=True)
            #generated = generated[:256]
            
            #this one starts from scratch with no prompt
            generated = [0] #start the generated stuff by specifying channel 1
            while len(generated) <= prm['GEN_LENGTH']:
                n_output = tasksolver(tf.expand_dims(generated,axis=0)[0:1,-prm['N_TIMESTEPS']:])
                result = tf.math.top_k(n_output[0,-1], k=prm['TOP_K'], sorted=False)
                rv = result.values
                rv = tf.nn.softmax(result.values).numpy()
                ri = result.indices.numpy()
                choice = np.random.choice(ri, p=rv)
                generated.append(choice)
                
            midisave.save_midi(generated, f"outs/{training_sess_id}_{steps}.mid")


        if steps % prm['SAVE_STEPS'] == 0:
            checkpoint = tf.train.Checkpoint(opt=optimizer, model=tasksolver)
            ckptfolder = checkpoint.save(file_prefix=prm['MODEL_SAVE_DIR']+'/weights/ckpt')
            
            sets = {
                'steps':steps+1, 
                'folder':ckptfolder, 
                'training_sess_id': training_sess_id
            }
            
            jsonsave(sets, prm['MODEL_SAVE_DIR']+'/settings.txt')
            
            print("Saved.", end="    \r")

        print()

        all_losses = []
        all_accs_char = []
        
    steps += 1
