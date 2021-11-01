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
    
def set_gpu_settings():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

set_gpu_settings()

import functools
import time
import enum
import matplotlib.pyplot as plt
import datahandler
import midisave

from counting import Counting
from transformer import Transformer, DenseLayer, EmbeddingLayer, Normalizer, floattype
import posenc

def get_replacer(tokens):
    r = np.arange(len(tokens))
    np.random.shuffle(r)
    new_tokens = tokens[r]
    replacer = np.arange(16)
    replacer[tokens] = new_tokens
    return replacer
    
def replace_tokenchannel(d, replacer):
    real = d-midisave.TOKEN_CHANNEL_PROGRAM
    inst = real//16
    chan = real%16
    chan = replacer[chan]
    total = chan+inst*16
    d = np.where(np.logical_and(np.less_equal(midisave.TOKEN_CHANNEL_PROGRAM,d),np.less(d,midisave.TOKEN_CHANNEL_PROGRAM+16*128)), total, d)
    return d

def augment_channels(d1, d2):
    tokens = np.array([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15])
    replacer = get_replacer(tokens)
    d1 = replace_tokenchannel(d1, replacer)
    d2 = replace_tokenchannel(d2, replacer)
    return d1, d2

def make_replacer(add, start, end):
    tokens = np.arange(end-start,dtype=np.int64)
    tokens = np.clip(tokens+add, 0, end-start-1)
    tokens += start
    prefix = np.arange(0,start)
    postfix = np.arange(end,midisave.N_TOKEN_TOTAL)
    replacer = np.concatenate([prefix, tokens, postfix],axis=-1)
    assert replacer.shape[0] == midisave.N_TOKEN_TOTAL, f"Note replacer size is wrong. should be {midisave.N_TOKEN_TOTAL} but is {replacer.shape}"
    return replacer

def make_note_on_replacer(add):
    replacer = make_replacer(add, midisave.TOKEN_NOTE_ON, midisave.TOKEN_NOTE_ON+128)
    return replacer
    
def make_note_off_replacer(add):
    replacer = make_replacer(add, midisave.TOKEN_NOTE_OFF, midisave.TOKEN_NOTE_OFF+128)
    return replacer

def make_velocity_replacer(add):
    replacer = make_replacer(add, midisave.TOKEN_VELOCITY+1, midisave.TOKEN_VELOCITY+128)
    return replacer
    
def make_cv_replacer(add):
    replacer = make_replacer(add, midisave.TOKEN_CONTROLLER_VALUE+0, midisave.TOKEN_CONTROLLER_VALUE+128)
    return replacer
    
def make_replacers(f, range_value):
    ret = []
    
    for num in range(-range_value, range_value+1):
        ret.append(f(num))
    return ret

note_on_replacers = make_replacers(make_note_on_replacer, 3)
note_off_replacers = make_replacers(make_note_off_replacer, 3)

def augment_notes(d1, d2):
    r_id = random.randint(0, len(note_on_replacers)-1)
    replacer_on = note_on_replacers[r_id]
    replacer_off = note_off_replacers[r_id]
    
    d1 = replacer_on[d1]
    d1 = replacer_off[d1]
    d2 = replacer_on[d2]
    d2 = replacer_off[d2]
    
    return d1, d2

velocity_replacers = make_replacers(make_velocity_replacer, 16)
def augment_velocities(d1, d2):
    r_id = random.randint(0, len(velocity_replacers)-1)
    replacer = velocity_replacers[r_id]
    d1 = replacer[d1]
    d2 = replacer[d2]
    return d1, d2

cv_replacers = make_replacers(make_cv_replacer, 16)
def augment_cvs(d1, d2):
    r_id = random.randint(0, len(cv_replacers)-1)
    replacer = cv_replacers[r_id]
    d1 = replacer[d1]
    d2 = replacer[d2]
    return d1, d2

if len(sys.argv) != 2:
    print("Give log save name as argument.")
    exit(0)
if not os.path.exists("outs"):
    os.mkdir("outs")

def jsonload(filename):
    return json.load(open(filename,"r"))
    
def jsonsave(obj, filename):
    json.dump(obj, open(filename, "w"))
    
prm = {} #parameters

generate = False
if generate:
    prm['PRINT_STEPS'] = 1
    prm['VALIDATION_STEPS'] = prm['PRINT_STEPS']*80000000
    prm['GEN_STEPS'] = prm['PRINT_STEPS']
    prm['SAVE_STEPS'] = prm['PRINT_STEPS']*3200000
else:
    prm['PRINT_STEPS'] = 64
    prm['VALIDATION_STEPS'] = prm['PRINT_STEPS']*8#0000000
    prm['GEN_STEPS'] = prm['PRINT_STEPS']*16000000
    prm['SAVE_STEPS'] = prm['PRINT_STEPS']*32#00000
prm['MODEL_SIZE'] = 576
prm['XF_LAYERS'] = 8
prm['XF_HEADS'] = 9
prm['LEARNING_RATE'] = 0.001
prm['LEARNING_RATE_MIN'] = 0.001
prm['ADAM_EPSILON'] = 1e-5
prm['BATCH_SIZE'] = 16
prm['WARMUP_STEPS'] = 0
prm['N_TIMESTEPS'] = 64
prm['VOCAB_SIZE'] = midisave.N_TOKEN_TOTAL
prm['GEN_LENGTH'] = 4096
prm['TOP_P'] = 0.92
prm['MODEL_SAVE_DIR'] = "Q:\\midi_model_saves\\" + sys.argv[1]
prm['TOKEN_SHIFT'] = 4
prm['AUX_LOSSES'] = False
prm['EVALUATION_ITERS'] = 6
prm['EMBEDDING_SLICER'] = 1
#prm['POSEMB_SIZE'] = prm['N_TIMESTEPS']
prm['GRADIENT_ACCUMULATION'] = None
prm['VALIDATION_LENGTHS'] = [64, 128, 256, 512, 1024]

tf.keras.mixed_precision.set_global_policy('mixed_float16')

for key in prm:
    print(f"{key}={prm[key]} ",end="")
print()

if not os.path.exists(prm['MODEL_SAVE_DIR']):
    os.mkdir(prm['MODEL_SAVE_DIR'])

steps = 1
tokens_seen = prm['N_TIMESTEPS']*prm['BATCH_SIZE']

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


#data = datahandler.NumpyFileShuffle(prm['N_TIMESTEPS'], filename_t=f"C:\\datasets\\midi\\training_tokens_{midisave.version}.npy", filename_v=f"C:\\datasets\\midi\\validation_tokens_{midisave.version}.npy", vocab_size=prm['VOCAB_SIZE'])

data = datahandler.NumpyFileShuffle(prm['N_TIMESTEPS'], filename_t=f"C:\\datasets\\midi\\lakh_training_tokens_v4.npy", filename_v=f"C:\\datasets\\midi\\validation_tokens_{midisave.version}.npy", vocab_size=prm['VOCAB_SIZE'])


#data = datahandler.NumpyFileShuffle(prm['N_TIMESTEPS'], filename_t=f"C:\\datasets\\midi\\gamemidi_tokens_v5.npy", filename_v=f"C:\\datasets\\midi\\validation_tokens_{midisave.version}.npy", vocab_size=prm['VOCAB_SIZE'])

#unique, counts = np.unique(data.tokens_t, return_counts=True)
#print(list(zip(unique,counts)))
#exit(0)

def distance_from_uniform(x):
    N = x.shape[-1]
    return tf.reduce_sum(tf.math.xlogy(x, x), axis=-1) + tf.math.log(N) * tf.reduce_sum(x, axis=-1)

class MidiEmbedder(tf.keras.Model):
    def __init__(self):
        super(MidiEmbedder,self).__init__()
        self.embedding_slicer = prm['EMBEDDING_SLICER']
    
        self.chans_emb = EmbeddingLayer(16, prm['MODEL_SIZE']//2//self.embedding_slicer)
        self.insts_emb = EmbeddingLayer(128, prm['MODEL_SIZE']//2//self.embedding_slicer)
        self.rest_emb = EmbeddingLayer(prm['VOCAB_SIZE']-16*128, prm['MODEL_SIZE']//2//self.embedding_slicer)
        
    def call(self, n_input):
        chans = n_input%16
        insts = tf.math.minimum(n_input//16,127)
        rest = tf.math.maximum(n_input-16*128,0)
        
        e_chans = self.chans_emb(chans)
        e_insts = self.insts_emb(insts)
        e_rest = self.rest_emb(rest)
        
        which = tf.expand_dims(n_input<16*128, axis=-1)
        ret = tf.where(which, e_chans+e_insts, e_rest)
        
        return ret

class MidiLogits(tf.keras.Model):
    def __init__(self):
        super(MidiLogits,self).__init__()
        
        self.chans_logits = DenseLayer(16)
        self.insts_logits = DenseLayer(128)
        self.rest_logits = DenseLayer(prm['VOCAB_SIZE']-16*128)
        
    def call(self, n_input):
        e_chans = self.chans_logits(n_input)
        e_insts = self.insts_logits(n_input)
        e_rest = self.rest_logits(n_input)
        
        e_chans = tf.expand_dims(e_chans, axis=-2)
        e_insts = tf.expand_dims(e_insts, axis=-1)
        
        e_insts = tf.broadcast_to(e_insts, [e_insts.shape[0], e_insts.shape[1], 128, 16])
        
        e_chans_insts = e_chans+e_insts
        
        e_chans_insts = tf.reshape(e_chans_insts, [e_chans_insts.shape[0], e_chans_insts.shape[1], e_chans_insts.shape[2]*e_chans_insts.shape[3]])
        
        ret = tf.concat([e_chans_insts,e_rest],axis=-1)
        
        return ret

class TaskSolverRecursive(tf.keras.Model):
    def __init__(self):
        super(TaskSolverRecursive,self).__init__()
        
        self.embedding_slicer = prm['EMBEDDING_SLICER']

        #self.embedding = EmbeddingLayer(prm['VOCAB_SIZE'], prm['MODEL_SIZE']//2//self.embedding_slicer)
        self.embedding = MidiEmbedder()
        
        #self.posemb = EmbeddingLayer(prm['POSEMB_SIZE'], prm['MODEL_SIZE'])

        self.xformer = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=True, use_counter=False, normalize_columns=False, aux_losses=True, token_shift_n=prm['TOKEN_SHIFT'])
        #self.xformer = Transformer(model_size=prm['MODEL_SIZE'], num_layers=1, heads=prm['XF_HEADS'], use_causal_mask=True, use_counter=False, normalize_columns=False, aux_losses=True, token_shift_n=prm['TOKEN_SHIFT'])
        
        
        self.time_data_multiplier = self.add_weight(f'time_data_mult', shape=[], initializer=tf.constant_initializer(value=1.0), trainable=False)
        
        normalizer = functools.partial(tf.keras.layers.LayerNormalization, center=False, scale=True)
        self.output_norm = normalizer()
        self.logit_outputs = DenseLayer(prm['VOCAB_SIZE'], use_bias=False)
        #self.logit_outputs = MidiLogits()
        
    def call(self, n_input, time_data, is_training, n_iters=prm['XF_LAYERS']):
        n_output = n_input
        n_output = self.embedding(n_output)
        
        
        
        n_output = tf.expand_dims(n_output,axis=-1)
        time_data = time_data*self.time_data_multiplier
        time_data = tf.expand_dims(time_data,axis=-1)
        n_output = tf.concat([n_output,time_data],axis=-1)

        n_output = tf.reshape(n_output, [n_output.shape[0], n_output.shape[1], prm['MODEL_SIZE']])
        
        #posemb here
        #pmb = tf.range(n_input.shape[-1], dtype=tf.int64) 
        #pmb = tf.broadcast_to(tf.expand_dims(pmb, axis=0), [n_input.shape[0],pmb.shape[-1]])
        #if is_training:
        #    pmb += tf.random.uniform(shape=[n_input.shape[0],1], minval=0, maxval=prm['POSEMB_SIZE'], dtype=tf.int64)
        
        #pmb %= prm['POSEMB_SIZE']
        
        #pmb = self.posemb(pmb)
        #n_output += pmb
        
        
        if self.embedding_slicer > 1:
            n_output = tf.reshape(tf.repeat(tf.expand_dims(n_output,axis=-1), self.embedding_slicer, axis=-1),[n_output.shape[0],n_output.shape[1],-1])
        
        outputs = []
        n_output = self.xformer(n_output)[-1]
        logits = n_output
        logits = self.output_norm(logits)
        logits = self.logit_outputs(logits)
        return logits


    def call_gen(self, n_input, transformer_cache=None, tokenshift_cache=None):
        n_output = n_input
        n_output = self.embedding(n_output)
        
        if self.embedding_slicer > 1:
            n_output = tf.reshape(tf.repeat(tf.expand_dims(n_output,axis=-1), self.embedding_slicer, axis=-1),[n_output.shape[0],n_output.shape[1],-1])

        outputs = []
        new_tf_caches = []
        new_tokenshift_caches = []
        for n in range(n_iters):
            n_output = self.xformer
            n_output, new_cache, _, new_tokenshift_cache = self.xformer.call_gen(n_output, cache_tgt=transformer_cache, cache_token_shift=tokenshift_cache)
            new_tf_caches.append(new_tf_cache)
            new_tokenshift_caches.append(new_tokenshift_cache)
        n_output = self.logit_outputs(n_output)
        return n_output, new_tf_caches, new_tokenshift_caches

class TaskSolver(tf.keras.Model):
    def __init__(self):
        super(TaskSolver,self).__init__()

        self.embedding = EmbeddingLayer(prm['VOCAB_SIZE'], prm['MODEL_SIZE'])

        self.xformer = Transformer(model_size=prm['MODEL_SIZE'], num_layers=prm['XF_LAYERS'], heads=prm['XF_HEADS'], use_causal_mask=True, use_counter=False, normalize_columns=False, aux_losses=True, token_shift_n=prm['TOKEN_SHIFT'])
        
        #normalizer = functools.partial(tf.keras.layers.LayerNormalization, center=False, scale=True)
        #self.output_norm = [normalizer() for _ in range(prm['XF_LAYERS'])]
        self.logit_outputs = [DenseLayer(prm['VOCAB_SIZE']) for _ in range(prm['XF_LAYERS'])]
        
        
    def call(self, n_input):
        n_output = n_input
        n_output = self.embedding(n_output)
        
        n_output = self.xformer(n_output)
        
        outputs = []
        for n in range(prm['XF_LAYERS']):
            logits = n_output[n]
            if (not prm['AUX_LOSSES']) and n < prm['XF_LAYERS']-1:
                logits = tf.stop_gradient(logits)
            #logits = self.logit_outputs[n](self.output_norm[n](logits))
            logits = self.logit_outputs[n](logits)
            outputs.append(logits)
            
        return outputs

    def call_gen(self, n_input, transformer_cache=None, tokenshift_cache=None):
        n_output = n_input
        n_output = self.embedding(n_output)
        
        n_output, new_cache, _, new_tokenshift_cache = self.xformer.call_gen(n_output, cache_tgt=transformer_cache, cache_token_shift=tokenshift_cache)
        n_output = self.logit_outputs(n_output)
        return n_output, new_cache, new_tokenshift_cache
        
tasksolver = TaskSolverRecursive()

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
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

start_time = time.time()

@tf.function
def do_step(n_input, target, training, time_data, n_iters=prm['XF_LAYERS']):
    losses = []
    accuracies_c = []
    if training:
        with tf.GradientTape() as tape:
            n_output = tasksolver(n_input, time_data, is_training=training)
            loss_c = tf.keras.losses.sparse_categorical_crossentropy(target, tf.cast(n_output, tf.float32), from_logits=True)
            loss_c = tf.reduce_mean(loss_c, axis=[-1,-2])
            pred_ids = tf.keras.backend.argmax(n_output,axis=-1)
            loss_scaled = optimizer.get_scaled_loss(loss_c)
            losses.append(loss_c)
        gradients = tape.gradient([loss_scaled], tasksolver.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, tasksolver.trainable_variables))
    else:
        n_output = tasksolver(n_input, time_data, is_training=training, n_iters=n_iters)
        loss_c = tf.keras.losses.sparse_categorical_crossentropy(target, tf.cast(n_output, tf.float32), from_logits=True)
        loss_c = tf.reduce_mean(loss_c, axis=[-1,-2])
        pred_ids = tf.keras.backend.argmax(n_output,axis=-1)
        losses.append(loss_c)

    correct_chars = tf.reduce_mean(tf.cast(tf.equal(pred_ids,target),dtype=tf.float32),axis=-1)
    accuracy_c = tf.squeeze(tf.reduce_mean(correct_chars, axis=-1))
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
    if 'tokens_seen' in settings:
        tokens_seen = settings['tokens_seen']
    else:
        tokens_seen = steps*prm['N_TIMESTEPS']*prm['BATCH_SIZE']

with open("training_log.txt", "a", encoding='utf8') as myfile:
    write_header(myfile)

def get_time_embedding(n_input, randomize=True):
    freqs = np.power(2.0, -np.linspace(-7.0,0.0,prm['MODEL_SIZE']//4)).astype(np.float32)
    freqs *= 3.141592653589793*2.0
    
    #n_input = n_input.astype(np.int64)
    
    #print(n_input.shape, n_input.dtype)
    #print(midisave.token_to_ms.shape, midisave.token_to_ms.dtype)
    
    times = midisave.token_to_ms[n_input]
    times = tf.math.cumsum(times)
    
    if randomize:
        times *= tf.random.uniform([], minval=0.9, maxval=1.1)
    
    timfreq = times[:,None]*freqs[None,:]
    
    times1 = tf.math.cos(timfreq)
    times2 = tf.math.sin(timfreq)
    times = tf.concat([times1,times2], axis=-1)*0.06
    times = tf.cast(times, floattype)
    return times
    

param_shown = False
while True:
    input_data = []
    targets = []
    time_data = []
    for _ in range(prm['BATCH_SIZE']):
        n_input, target = data.get_random_t_data(prm['N_TIMESTEPS'])
        n_input, target = augment_channels(n_input, target)
        n_input, target = augment_notes(n_input, target)
        n_input, target = augment_velocities(n_input, target)
        n_input, target = augment_cvs(n_input, target)
        
        times = get_time_embedding(n_input)
        
        input_data.append(n_input)
        targets.append(target)
        time_data.append(times)
        
    input_data = tf.stack(input_data,axis=0)
    targets = tf.stack(targets,axis=0)
    time_data = tf.stack(time_data,axis=0)
    
    input_data = tf.cast(input_data, tf.int64)
    targets = tf.cast(targets, tf.int64)
    
    losses, accuracies_char, _ = do_step(input_data, target=targets, time_data=time_data, training=True)
    all_losses.extend(losses)
    all_accs_char.extend(accuracies_char)
    
    if not param_shown:
        param_shown = True
        print(f"#params = {tf.reduce_sum([tf.size(tens) for tens in tasksolver.trainable_variables])}")
        
        #plot posemb

        #pmb = tf.range(prm['POSEMB_SIZE'], dtype=tf.int64)
        #pmb = tasksolver.posemb(pmb[None])[0]

        #pmb = tf.range(0,midisave.N_TOKEN_TOTAL-16*128, dtype=tf.int64)
        #pmb = tf.range(0,384, dtype=tf.int64)
        #pmb = tf.convert_to_tensor([midisave.convert_delay(x) for x in range(3072)], dtype=tf.int64)
        
        #pmb = tasksolver.embedding.rest_emb(pmb[None])[0]

        #pmb = tf.range(0,16, dtype=tf.int64)
        #pmb = tasksolver.embedding.chans_emb(pmb[None])[0]

        #pmb = tf.range(0,128, dtype=tf.int64)
        #pmb = tasksolver.embedding.insts_emb(pmb[None])[0]

        #pmb = tf.math.l2_normalize(pmb)
        #pmb = tf.einsum('ab,cb->ac',pmb,pmb)
        #pmb *= (1.0-tf.eye(pmb.shape[-1]))
        #plt.imshow(pmb)
        #plt.show()
        #exit(0)
        

        

    if steps % prm['PRINT_STEPS'] == 0:
        def format_to(stuff):
            return str(stuff).replace('[','').replace(']','') + ' | '
    
        with open("soft_attention.txt", "a", encoding='utf8') as myfile:
            myfile.write(str(steps)+ " ")
            """for ta in tasksolver.xformer.tgt_alpha:
                myfile.write(str(ta.numpy()).replace('[','').replace(']',''))
            myfile.write(" | ")
            for ta in tasksolver.xformer.tgt_beta:
                myfile.write(str(ta.numpy()).replace('[','').replace(']',''))
            myfile.write(" | ")
            for ta in tasksolver.xformer.ffn_alpha:
                myfile.write(str(ta.numpy()).replace('[','').replace(']',''))
            myfile.write(" | ")
            for ta in tasksolver.xformer.ffn_beta:
                myfile.write(str(ta.numpy()).replace('[','').replace(']',''))
            myfile.write(" | ")"""
            
            myfile.write(format_to(tasksolver.time_data_multiplier.numpy()))
            #myfile.write(format_to(tasksolver.posemb.norm.mult.numpy()))
            myfile.write("\n")



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
            myfile.write(f"{steps} {tokens_seen} {round(totaltime,4)} s ")
            myfile.write(f"{round(token_per_second*86400.0/1000.0/1000.0/1000.0,4)} Gtok/day ")
            myfile.write(f"l {cts2(losses)} ")
            myfile.write(f"a {cts2(accs_char)} ")
            myfile.write(f"\n")

        print(f"{steps} {tokens_seen} {round(totaltime,4)} s ", end='')
        print(f"{round(token_per_second*86400.0/1000.0/1000.0/1000.0,4)} Gtok/day ", end='')
        print(f"l {remove_braces(str(losses))} ", end='')
        print(f"a {remove_braces(str(accs_char))} ", end='')
        print(f"lr {lr_scheduler()} ", end='')
        

        if steps % prm['VALIDATION_STEPS'] == 0:
            vdata = data.get_v_data()
            
            print(f"vl/a ", end="")
            
            #testlengths = [prm['N_TIMESTEPS'], prm['N_TIMESTEPS']*2, prm['N_TIMESTEPS']*4]
            testlengths = prm['VALIDATION_LENGTHS']
            for testlength in testlengths:
            
                vdata_input = vdata[:-1]
                vdata_target = vdata[1:]
                
                vdata_input = vdata_input[:vdata_input.shape[0]-vdata_input.shape[0]%testlengths[-1]]
                vdata_input = np.reshape(vdata_input, [-1, testlength])
                vdata_input = tf.cast(vdata_input, tf.int64)

                vdata_target = vdata_target[:vdata_target.shape[0]-vdata_target.shape[0]%testlengths[-1]]
                vdata_target = np.reshape(vdata_target, [-1, testlength])
                vdata_target = tf.cast(vdata_target, tf.int64)
                
                #v_result = np.zeros(shape=(2,vdata_input.shape[0],prm['EVALUATION_ITERS']), dtype=np.float32)
                v_result = np.zeros(shape=(2,vdata_input.shape[0],1), dtype=np.float32)
                
                for start in range(0,vdata_input.shape[0]):
                    time_data = get_time_embedding(vdata_input[start], randomize=False)[None]
                    losses, accuracies, _ = do_step(vdata_input[start:start+1], target=vdata_target[start:start+1], time_data=time_data, training=False, n_iters=prm['EVALUATION_ITERS'])
                    v_result[0,start] = losses[0].numpy()
                    v_result[1,start] = accuracies[0].numpy()
                    
                v_result = list(np.mean(v_result, axis=-2))
                
                print(f"{remove_braces(str(v_result[0]))} ", end="")
                print(f"{remove_braces(str(v_result[1]))} ", end="")
                    
                with open("validation_log.txt", "a", encoding='utf8') as myfile:
                    myfile.write(f"steps {steps} length {testlength} ")
                    myfile.write(f"l {remove_braces(str(v_result[0]))} ")
                    myfile.write(f"a {remove_braces(str(v_result[1]))} ")

            with open("validation_log.txt", "a", encoding='utf8') as myfile:
                myfile.write(f"\n")


        if steps % prm['GEN_STEPS'] == 0:
            notes_on = []
            for c in range(16):
                notes_on.append([])
            current_channel = 0
            current_instrument = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            current_note = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            
            #this one loads a midi 
            #generated = midisave.load_midi("C:\\datasets\\midi\\validation_midis\\spora.mid", clip=True)
            #generated = midisave.load_midi("C:\\datasets\\midi\\omat\\dehuman.mid", clip=True)
            #generated = midisave.load_midi("C:\\datasets\\midi\\omat\\piano001.mid", clip=True)
            #generated = midisave.load_midi("C:\\datasets\\midi\\validation_midis\\main.mid", clip=True)
            generated = midisave.load_midi("Q:\\gamemidi\\Doom\\02 - At Doom's Gate (E1M1).mid", clip=True)
            #generated = midisave.load_midi("Q:\\gamemidi\\Descent\\Game01.mid", clip=True)
            #generated = midisave.load_midi("Q:\\gamemidi\\Rise of the Triad\\ROTT_005.mid", clip=True)
            #generated = midisave.load_midi("C:\\Users\\liimaeve\\wind.mid", clip=True)
            generated = generated[:384]
            #generated = [midisave.TOKEN_DELAY+16] #start the generated stuff with a short pause
            
            #TESTING START
            """n_output_old = tasksolver(tf.expand_dims(generated,axis=0)[0:1,-prm['N_TIMESTEPS']:])
            tf_ch = None
            tokenshift_ch = None
            nns = []
            for i in range(0,16):
                gen = generated[i:i+1]
                nn, tf_ch, tokenshift_ch = tasksolver.call_gen(tf.expand_dims(gen,axis=0), tf_ch, tokenshift_ch)
                nns.append(nn)
            n_output_new = tf.concat(nns, axis=-2)
            print(tf.reduce_mean(tf.square(n_output_new-n_output_old),axis=-1))
            #TESTING END
            exit(0)"""
            
            #generate the cache for the prompt
            if len(generated) > 1:
                #_, transformer_cache, tokenshift_cache = tasksolver.call_gen(tf.expand_dims(generated,axis=0)[0:1,:-1])
                pass
            else:
                transformer_cache = None
                tokenshift_cache = None
            print()
            
            graph_drawn = False
            while len(generated) <= prm['GEN_LENGTH']:
                #print(f"{len(generated)} ", flush=True, end="")
                #n_output,transformer_cache,tokenshift_cache = tasksolver.call_gen(tf.expand_dims(generated,axis=0)[0:1,-1:], transformer_cache, tokenshift_cache)
                t1 = tf.expand_dims(generated,axis=0)[0:1,-512:]
                time_data = get_time_embedding(t1[0].numpy(), randomize=False)[None]
                n_output = tasksolver(t1, time_data, is_training=False)
                
                #if not graph_drawn:
                """if True:
                    #plt.bar(np.arange(n_output[0,-1].shape[-1]), n_output[0,-1])
                    plt.plot(tf.nn.softmax(n_output[0,-1]).numpy())
                    #plt.imshow(n_output[0])
                    plt.show()                    
                    graph_drawn = True
                    if len(generated) > 32:
                        exit(0)"""

                #top-k sampling
                #result = tf.math.top_k(n_output[0,-1], k=prm['TOP_K'], sorted=False)
                #rv = result.values
                #rv = tf.nn.softmax(result.values).numpy()
                #ri = result.indices.numpy()
                #choice = int(np.random.choice(ri, p=rv))
                
                #top-p sampling
                result = tf.nn.softmax(n_output[0,-1]).numpy()

                sort_order = np.argsort(result)[::-1]
                result = np.sort(result)[::-1]
                max_index = np.sum(np.less(np.cumsum(result), prm['TOP_P']).astype(np.int64))
                max_index = np.maximum(1, max_index)
                sort_order = sort_order[:max_index]
                result = result[:max_index]
                result /= np.sum(result)
                choice = int(np.random.choice(sort_order, p=result))
                
                generated.append(choice)
            
                if midisave.TOKEN_CHANNEL_PROGRAM <= choice < midisave.TOKEN_CHANNEL_PROGRAM+16*128:
                    current_channel = (choice-midisave.TOKEN_CHANNEL_PROGRAM)%16
                    current_instrument[current_channel] = (choice-midisave.TOKEN_CHANNEL_PROGRAM)//16
                
                elif midisave.TOKEN_NOTE_ON <= choice < midisave.TOKEN_NOTE_ON+128:
                    current_note[current_channel] = choice-midisave.TOKEN_NOTE_ON
                elif midisave.TOKEN_VELOCITY <= choice < midisave.TOKEN_VELOCITY+128:
                    if current_note[current_channel] != -1:
                        notes_on[current_channel].append((current_note[current_channel],len(generated)))
                        current_note[current_channel] = -1
                elif midisave.TOKEN_NOTE_OFF <= choice < midisave.TOKEN_NOTE_OFF+128:
                    notes_on[current_channel] = [x for x in notes_on[current_channel] if x[0] != choice-midisave.TOKEN_NOTE_OFF]
                    
                for c in range(16):
                    if c==9: #don't do this for the drum channel
                        continue
                    bad_notes = [x for x in notes_on[c] if x[1]<len(generated)-prm['N_TIMESTEPS']]
                    notes_on[c] = [x for x in notes_on[c] if x[1]>=len(generated)-prm['N_TIMESTEPS']]
                    
                    #if len(bad_notes) > 0:
                        #print(f"{len(generated)} c{c} {bad_notes} bad notes.")
                    
                    for bad_note in bad_notes:
                        #print(f"{len(generated)} BAD NOTE c{c} n{bad_note[0]} t{bad_note[1]}")
                        if current_channel != c:
                            generated.append(midisave.TOKEN_CHANNEL_PROGRAM+c+current_instrument[c]*16)
                            #n_output,transformer_cache,tokenshift_cache = tasksolver.call_gen(tf.expand_dims(generated,axis=0)[0:1,-1:], transformer_cache, tokenshift_cache)
                            current_channel = c
                            
                        generated.append(midisave.TOKEN_NOTE_OFF+bad_note[0])
                        #n_output,transformer_cache,tokenshift_cache = tasksolver.call_gen(tf.expand_dims(generated,axis=0)[0:1,-1:], transformer_cache, tokenshift_cache)
            
            midisave.save_midi(generated, f"outs/{training_sess_id}_{steps}_topp{prm['TOP_P']}_{str(random.randrange(1000000000))}.mid")


        if steps % prm['SAVE_STEPS'] == 0:
            checkpoint = tf.train.Checkpoint(opt=optimizer, model=tasksolver)
            ckptfolder = checkpoint.save(file_prefix=prm['MODEL_SAVE_DIR']+'/weights/ckpt')
            
            sets = {
                'steps':steps+1, 
                'tokens_seen':tokens_seen+prm['N_TIMESTEPS']*prm['BATCH_SIZE'],
                'folder':ckptfolder, 
                'training_sess_id': training_sess_id
            }
            
            jsonsave(sets, prm['MODEL_SAVE_DIR']+'/settings.txt')
            
            print("Saved.", end="    \r")

        print()

        all_losses = []
        all_accs_char = []
        
    steps += 1
    tokens_seen += prm['N_TIMESTEPS']*prm['BATCH_SIZE']
