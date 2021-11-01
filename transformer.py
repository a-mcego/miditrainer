import numpy as np
import tensorflow as tf
import functools

import counting

floattype = tf.float16

def reduce(matrix, axis):
    mean = tf.math.reduce_mean(matrix, axis=axis, keepdims=True)
    stddev = tf.math.reduce_std(matrix, axis=axis, keepdims=True)
    return (matrix - mean)/(stddev + 1e-2)

class LayerScaler(tf.keras.layers.Layer):
    def __init__(self, initial_value=1.0):
        super(LayerScaler, self).__init__()
        self.mult = self.add_weight('mult', shape=[], initializer=tf.constant_initializer(value=initial_value), trainable=True)
        self.ln = tf.keras.layers.LayerNormalization(center=False, scale=True)
    
    def call(self, data):
        return self.ln(data)*self.mult

class ScaledEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(ScaledEmbedding, self).__init__()

        #this just replaces the normal embedding with a equalized LR one
        self.embedding = tf.keras.layers.Embedding(input_dim, output_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0))
        self.norm = LayerScaler(initial_value=1.0) #=0.05)
        
    def call(self, data):
        return self.norm(self.embedding(data))

class ScaledDense(tf.keras.layers.Layer):
    def __init__(self, model_size, use_bias=True, activation=None):
        super(ScaledDense, self).__init__()
        self.activation = activation
        self.use_bias = use_bias
        self.model_size = model_size
        if use_bias:
            self.bias = self.add_weight('bias', shape=[model_size], initializer=tf.zeros_initializer(), trainable=True)
        
    def build(self, input_shape):
        #this is the glorot normal init: stddev = sqrt(2.0 / (fan_in + fan_out))
        #self.dense_weights = self.add_weight('dense_weights', shape=[input_shape[-1], self.model_size], initializer=tf.keras.initializers.TruncatedNormal(stddev=1.0,mean=0.0), trainable=True)
        #mult_init = np.sqrt(2.0 / (input_shape[-1] + self.model_size)) 
        #self.mult = self.add_weight('mult', shape=[], initializer=tf.constant_initializer(value=mult_init), trainable=False)
        
        #this is the glorot uniform init: limit = sqrt(6.0 / (fan_in + fan_out))
        self.dense_weights = self.add_weight('dense_weights', shape=[input_shape[-1], self.model_size], initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0), trainable=True)
        mult_init = np.sqrt(6.0 / (input_shape[-1] + self.model_size)) 
        self.mult = self.add_weight('mult', shape=[], initializer=tf.constant_initializer(value=mult_init), trainable=True)

    def call(self, input_data):
        ret = tf.matmul(input_data, self.dense_weights*self.mult)
        if self.use_bias:
            ret += self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        return ret

DenseLayer = tf.keras.layers.Dense
#DenseLayer = ScaledDense
#EmbeddingLayer = tf.keras.layers.Embedding
EmbeddingLayer = ScaledEmbedding

Normalizer = LayerScaler
#Normalizer = functools.partial(tf.keras.layers.LayerNormalization, center=False, scale=True)

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, n_heads, use_causal_mask):
        super(MultiHeadAttention, self).__init__()
        self.model_size = model_size
        self.head_size = model_size // n_heads
        self.n_heads = n_heads

        self.w_q = DenseLayer(model_size, use_bias=False)
        self.w_kv = DenseLayer(model_size*2, use_bias=False)

        self.wo = DenseLayer(model_size, use_bias=False)

        norm = 1.0/np.sqrt(np.array(self.head_size).astype(np.float32))
        self.normalizer = self.add_weight('sat_weight', shape=[], initializer=tf.constant_initializer(value=norm), trainable=True, dtype=floattype)
        
        self.use_causal_mask = use_causal_mask
        
        #head multipliers from normformer
        self.head_mults = self.add_weight('head_mults', shape=[1,1,n_heads,1], initializer=tf.constant_initializer(value=1.0), trainable=True)
        
    def call(self, query, value, mask=None):
        heads = []
        #B x T x 2*d_m
        out_kv = self.w_kv(value)
        #B x T x d_m
        out_q = self.w_q(query)
        
        #[B x T x d_m, B x T x d_m]
        kvs_tmp = tf.split(out_kv, 2, axis=-1)
        #B x T x d_m
        out_k = kvs_tmp[0]
        #B x T x d_m
        out_v = kvs_tmp[1]
        
        #out_v = sqrelu(out_v)
        
        #B x T_q x H x d_h
        out_q = tf.reshape(out_q, [out_q.shape[0], out_q.shape[1], self.n_heads, -1])
        #B x T_k x H x d_h
        out_k = tf.reshape(out_k, [out_k.shape[0], out_k.shape[1], self.n_heads, -1])
        #B x T_v x H x d_h
        out_v = tf.reshape(out_v, [out_v.shape[0], out_v.shape[1], self.n_heads, -1])
        
        #B x H x T_q x T_k
        score = tf.einsum('bqhd,bkhd->bhqk', out_q, out_k, optimize='optimal')*self.normalizer
        if mask is not None:
            score += mask[None,:,:,:]
        #B x H x T_q x T_ks
        alignment = tf.nn.softmax(score, axis=-1)

        #if mask is not None:
        #    score *= mask[None,:,:,:]
        #B x H x T_q x T_ks
        #alignment = tf.square(score)
        
        #B x T_q x H x d_h
        heads = tf.einsum('bhqk,bkhd->bqhd', alignment, out_v, optimize='optimal')
        heads = reduce(heads, axis=-2)
        heads *= self.head_mults #head multipliers from normformer
        
        #B x T_q x d_m
        heads = tf.reshape(heads, [query.shape[0],-1,self.model_size])
        heads = self.wo(heads)
        return heads

class ScaledLengthNormalization(tf.keras.Model):
    def __init__(self):
        super(ScaledLengthNormalization, self).__init__()
        self.scale = self.add_weight('scale', shape=[], initializer='ones', trainable=True)
        
    def call(self, data):
        return tf.math.l2_normalize(data, axis=-1, epsilon=1e-6)*self.scale


def token_shift(n_output, n):
    n_timesteps = n_output.shape[1]
    #n_output is B x T x D
    n_output = tf.reshape(n_output, [n_output.shape[0], n_output.shape[1], n, -1])
    
    #n_output is B x T+TS-1 x TS x D/TS
    n_output = tf.concat([tf.zeros(shape=[n_output.shape[0], n-1, n_output.shape[2], n_output.shape[3]], dtype=n_output.dtype), n_output], axis=1)
    
    out = []
    for i in range(n):
        out.append(n_output[:, i:i+n_timesteps, i, :])
    return tf.concat(out, axis=-1)

def sqrelu(x):
    return tf.square(tf.nn.relu(x))

#no positional encoding, no classification layers, no frills.
#basic transformer is all you need :D
#
#this version of transformer attends to the source sequence *first*
#only then to the target sequence
class Transformer(tf.keras.Model):
    def __init__(self, model_size, num_layers, heads, use_causal_mask, use_counter, normalize_columns, aux_losses=False, abstraction_size=None, token_shift_n=None):
        super(Transformer, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.heads = heads
        self.token_shift_n = token_shift_n
        self.aux_losses = aux_losses
        
        #normalizer = ScaledLengthNormalization
        #normalizer = tf.keras.layers.LayerNormalization
        #normalizer = functools.partial(tf.keras.layers.LayerNormalization, center=False, scale=True)
        normalizer = Normalizer

        self.src_att = [MultiHeadAttention(model_size, heads, use_causal_mask=False) for _ in range(num_layers)]
        self.src_att_norm = [normalizer() for _ in range(num_layers)]
        self.src_att_postnorm = [normalizer() for _ in range(num_layers)]

        self.tgt_att = [MultiHeadAttention(model_size, heads, use_causal_mask=use_causal_mask) for _ in range(num_layers)]
        self.tgt_att_norm = [normalizer() for _ in range(num_layers)]
        self.tgt_att_postnorm = [normalizer() for _ in range(num_layers)]
        
        self.dense_1 = [DenseLayer(model_size * 4, activation=sqrelu) for _ in range(num_layers)]
        self.dense_2 = [DenseLayer(model_size) for _ in range(num_layers)]
        
        #self.global_ffn = [GlobalFFN(model_size, heads, 64) for _ in range(num_layers)]
        self.ffn_norm = [normalizer() for _ in range(num_layers)]
        self.ffn_norm_mid = [normalizer() for _ in range(num_layers)]
        
        #self.tgt_alpha = [self.add_weight(f'alpha_tgt{n}', shape=[], initializer=tf.constant_initializer(value=1.0), trainable=False) for n in range(num_layers)]
        #self.ffn_alpha = [self.add_weight(f'alpha_ffn{n}', shape=[], initializer=tf.constant_initializer(value=1.0), trainable=False) for n in range(num_layers)]
        #self.tgt_beta = [self.add_weight(f'alpha_tgt{n}', shape=[], initializer=tf.constant_initializer(value=1.0), trainable=False) for n in range(num_layers)]
        #self.ffn_beta = [self.add_weight(f'alpha_ffn{n}', shape=[], initializer=tf.constant_initializer(value=1.0), trainable=False) for n in range(num_layers)]
        
        #self.kms = [KMeans(model_size, 8) for _ in range(num_layers)]
        
        self.use_causal_mask = use_causal_mask
        self.use_counter = use_counter
        
        if self.use_counter:
            assert abstraction_size is not None, "If use_counter==true, you must give an abstraction_size to the Transformer layer"
            self.abstraction_size = abstraction_size
            self.counting = [counting.Abstraction(model_size,self.abstraction_size, self.use_causal_mask) for _ in range(num_layers)]
            self.counting_norm = [normalizer() for _ in range(num_layers)]
            
        MAX_LENGTH = 3072+64
        bp = tf.linalg.band_part(tf.ones((1,MAX_LENGTH,MAX_LENGTH)), -1, 0)
        self.causal_mask = tf.where(bp==0,-50000000.0,0.0)
        #self.causal_mask = tf.where(bp==0,0.0,1.0)
        
        self.causal_mask = tf.cast(self.causal_mask, floattype)
        
    def call(self, target_sequence, source_sequence=None):
        if self.use_causal_mask:
            mask_size = target_sequence.shape[1]
            look_left_only_mask = self.causal_mask[:,:mask_size,:mask_size]
        else:
            look_left_only_mask = None
            
        src_att_in = target_sequence
        
        rets = []
        
        for i in range(self.num_layers):
            if self.token_shift_n is not None:
                src_att_in = token_shift(src_att_in,self.token_shift_n)
        
            #src_att_in has the data now
            
            if self.use_counter:
                src_att_in = self.counting[i](src_att_in)
                src_att_in = self.counting_norm[i](src_att_in)
        
            if source_sequence is not None:
                src_att_out = self.src_att_norm[i](src_att_in)
                src_att_out = self.src_att[i](src_att_out, source_sequence, None)
                src_att_out = src_att_in + self.src_att_postnorm[i](src_att_out)
            else:
                src_att_out = src_att_in
                
            tgt_att_in = src_att_out
            #tgt_att_in has the data now
            tgt_att_out = self.tgt_att_norm[i](tgt_att_in)
            tgt_att_out = self.tgt_att[i](tgt_att_out, tgt_att_out, look_left_only_mask)
            tgt_att_out = tgt_att_in + self.tgt_att_postnorm[i](tgt_att_out)
            #tgt_att_out = self.tgt_alpha[i]*tgt_att_in + self.tgt_beta[i]*tgt_att_out
                        
            ffn_in = tgt_att_out
            #ffn_in has the data now 
            ffn_mid = self.ffn_norm[i](ffn_in)
            ffn_mid = self.dense_1[i](ffn_mid)
            ffn_mid = self.ffn_norm_mid[i](ffn_mid)
            ffn_out = self.dense_2[i](ffn_mid)
            
            ffn_out = ffn_in + ffn_out
            #ffn_out = self.ffn_alpha[i]*ffn_in + self.ffn_beta[i]*ffn_out

            src_att_in = ffn_out
            #src_att_in has the data now
            
            if self.aux_losses or i+1==self.num_layers:
                rets.append(src_att_in)

        return rets
    
    def call_gen(self, target_sequence, source_sequence=None, cache_tgt=None, cache_src=None, cache_token_shift=None):
        if self.use_causal_mask:
            mask_size = target_sequence.shape[1]
            if cache_tgt is None:
                look_left_only_mask = self.causal_alibi_mask[:,:mask_size,:mask_size]
            else:
                cache_time = cache_tgt[0].shape[-2]
                look_left_only_mask = self.causal_alibi_mask[:,cache_time:cache_time+mask_size,:cache_time+mask_size]
        else:
            look_left_only_mask = None
            
        cache_src_new = []
        cache_tgt_new = []
        cache_token_shift_new = []
            
        src_att_in = target_sequence
        for i in range(self.num_layers):
            if self.token_shift_n is not None:
                if cache_token_shift is None:
                    cache_token_shift_new.append(src_att_in)
                    src_att_in = token_shift(src_att_in,self.token_shift_n)
                else:
                    original_len = src_att_in.shape[1]
                    new_token_cache = tf.concat([cache_token_shift[i], src_att_in],axis=-2)[:,-self.token_shift_n:,:]
                    cache_token_shift_new.append(new_token_cache)
                    src_att_in = token_shift(new_token_cache,self.token_shift_n)
                    src_att_in = src_att_in[:,-original_len:]
        
            #src_att_in has the data now
            
            if self.use_counter:
                src_att_in = self.counting[i](src_att_in)
                src_att_in = self.counting_norm[i](src_att_in)
        
            if source_sequence is not None:
                src_att_out = self.src_att_norm[i](src_att_in)
                src_att_out,new_cache = self.src_att[i].call_gen(src_att_out, source_sequence, None, cache_src[i] if cache_src is not None else None)
                cache_src_new.append(new_cache)
                src_att_out = src_att_in + src_att_out
            else:
                src_att_out = src_att_in
                
            tgt_att_in = src_att_out
            #tgt_att_in has the data now
            tgt_att_out = self.tgt_att_norm[i](tgt_att_in)
            tgt_att_out,new_cache = self.tgt_att[i].call_gen(tgt_att_out, tgt_att_out, look_left_only_mask, cache_tgt[i] if cache_tgt is not None else None)
            cache_tgt_new.append(new_cache)
            tgt_att_out = tgt_att_in + tgt_att_out
                        
            ffn_in = tgt_att_out
            #ffn_in has the data now 
            ffn_mid = self.ffn_norm[i](ffn_in)
            ffn_mid = self.dense_1[i](ffn_mid)
            
            ffn_out = self.dense_2[i](ffn_mid)

            ffn_out = ffn_out + ffn_in

            src_att_in = ffn_out
            #src_att_in has the data now

        return self.final_norm(src_att_in), cache_tgt_new, cache_src_new, cache_token_shift_new
    