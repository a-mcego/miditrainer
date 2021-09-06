import numpy as np
import tensorflow as tf

import counting

def reduce(matrix, axis):
    mean = tf.math.reduce_mean(matrix, axis=axis, keepdims=True)
    stddev = tf.math.reduce_std(matrix, axis=axis, keepdims=True)
    return (matrix - mean)/(stddev + 1e-2)

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, n_heads, use_causal_mask):
        super(MultiHeadAttention, self).__init__()
        self.model_size = model_size
        self.head_size = model_size // n_heads
        self.n_heads = n_heads

        self.w_q = tf.keras.layers.Dense(model_size)
        self.w_kv = tf.keras.layers.Dense(model_size*2)

        self.wo = tf.keras.layers.Dense(model_size)
        self.normalizer = 1.0/tf.math.sqrt(tf.dtypes.cast(self.head_size, tf.float32))
        
        self.use_causal_mask = use_causal_mask
        
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
        
        #B x T_q x H x d_h
        out_q = tf.reshape(out_q, [out_q.shape[0], out_q.shape[1], self.n_heads, -1])
        #B x T_k x H x d_h
        out_k = tf.reshape(out_k, [out_k.shape[0], out_k.shape[1], self.n_heads, -1])
        #B x T_v x H x d_h
        out_v = tf.reshape(out_v, [out_v.shape[0], out_v.shape[1], self.n_heads, -1])
        
        #B x H x T_q x T_k
        score = tf.einsum('bqhd,bkhd->bhqk', out_q, out_k)*self.normalizer
        
        if mask is not None:
            score += mask[None,:,:,:]
        
        #B x H x T_q x T_ks
        alignment = tf.nn.softmax(score, axis=-1)
        
        #B x T_q x H x d_h
        heads = tf.einsum('bhqk,bkhd->bqhd', alignment, out_v)

        #B x T_q x d_m
        heads = tf.reshape(heads, [query.shape[0],-1,self.model_size])
        heads = self.wo(heads)
        return heads

    #use this to generate with caching
    def call_gen(self, query, value, mask=None, cache=None):
        heads = []
        #B x T_kv x 2*d_m
        out_kv = self.w_kv(value)
        if cache is not None:
            out_kv = tf.concat([cache, out_kv], axis=-2)
        
        #B x T_q x d_m
        out_q = self.w_q(query)
        
        #[B x T x d_m, B x T x d_m]
        kvs_tmp = tf.split(out_kv, 2, axis=-1)
        #B x T x d_m
        out_k = kvs_tmp[0]
        #B x T x d_m
        out_v = kvs_tmp[1]
        
        #B x T_q x H x d_h
        out_q = tf.reshape(out_q, [out_q.shape[0], out_q.shape[1], self.n_heads, -1])
        #B x T_k x H x d_h
        out_k = tf.reshape(out_k, [out_k.shape[0], out_k.shape[1], self.n_heads, -1])
        #B x T_v x H x d_h
        out_v = tf.reshape(out_v, [out_v.shape[0], out_v.shape[1], self.n_heads, -1])
        
        #B x H x T_q x T_k
        score = tf.einsum('bqhd,bkhd->bhqk', out_q, out_k)*self.normalizer
        
        if mask is not None:
            score += mask[None,:,:,:]
        
        #B x H x T_q x T_ks
        alignment = tf.nn.softmax(score, axis=-1)
        
        #B x T_q x H x d_h
        heads = tf.einsum('bhqk,bkhd->bqhd', alignment, out_v)

        #B x T_q x d_m
        heads = tf.reshape(heads, [query.shape[0],-1,self.model_size])
        heads = self.wo(heads)
        return heads, out_kv

def token_shift(n_output, n):
    n_timesteps = n_output.shape[1]
    #n_output is B x T x D
    n_output = tf.reshape(n_output, [n_output.shape[0], n_output.shape[1], n, -1])
    
    #n_output is B x T+TS-1 x TS x D/TS
    n_output = tf.concat([tf.zeros(shape=[n_output.shape[0], n-1, n_output.shape[2], n_output.shape[3]], dtype=tf.float32), n_output], axis=1)
    
    out = []
    for i in range(n):
        out.append(n_output[:, i:i+n_timesteps, i, :])
    return tf.concat(out, axis=-1)

#no positional encoding, no classification layers, no frills.
#basic transformer is all you need :D
#
#this version of transformer attends to the source sequence *first*
#only then to the target sequence
class Transformer(tf.keras.Model):
    def __init__(self, model_size, num_layers, heads, use_causal_mask, use_counter, normalize_columns, abstraction_size=None, token_shift_n=None):
        super(Transformer, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.heads = heads
        self.token_shift_n = token_shift_n

        self.src_att = [MultiHeadAttention(model_size, heads, use_causal_mask=use_causal_mask) for _ in range(num_layers)]
        self.src_att_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        self.tgt_att = [MultiHeadAttention(model_size, heads, use_causal_mask=use_causal_mask) for _ in range(num_layers)]
        self.tgt_att_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation=tf.math.softplus) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        
        self.use_causal_mask = use_causal_mask
        
        self.final_norm = tf.keras.layers.LayerNormalization()
        
        self.use_counter = use_counter
        
        if self.use_counter:
            assert abstraction_size is not None, "If use_counter==true, you must give an abstraction_size to the Transformer layer"
            self.abstraction_size = abstraction_size
            self.counting = [counting.Abstraction(model_size,self.abstraction_size, self.use_causal_mask) for _ in range(num_layers)]
            self.counting_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
            
        MAX_LENGTH = 4096
        
        grid = np.mgrid[0:-MAX_LENGTH:-1, 0:MAX_LENGTH].astype(np.float32)
        grid = grid[0]+grid[1]
        #np.copy because it was read only for some reason, maybe because of np.mgrid[]
        grid = np.copy(np.broadcast_to(grid[None,:,:], [self.heads,MAX_LENGTH,MAX_LENGTH]))
        
        head_mults = np.power(2.0,-np.linspace(0.0, 8.0, num=heads+1)[1:])
        head_mults = head_mults[:, None, None]
        grid *= head_mults
        #grid *= 0.0 #ablation: remove ALIBI altogether
        
        self.causal_alibi_mask = tf.convert_to_tensor(grid)
        bp = tf.linalg.band_part(tf.ones((self.heads,MAX_LENGTH,MAX_LENGTH)), -1, 0)
        self.causal_alibi_mask = tf.where(bp==0,-90000000.0,self.causal_alibi_mask)
            
    def call(self, target_sequence, source_sequence=None):
        if self.use_causal_mask:
            mask_size = target_sequence.shape[1]
            look_left_only_mask = self.causal_alibi_mask[:,:mask_size,:mask_size]
        else:
            look_left_only_mask = None
            
        src_att_in = target_sequence
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
                src_att_out = src_att_in + src_att_out
            else:
                src_att_out = src_att_in
                
            tgt_att_in = src_att_out
            #tgt_att_in has the data now
            tgt_att_out = self.tgt_att_norm[i](tgt_att_in)
            tgt_att_out = self.tgt_att[i](tgt_att_out, tgt_att_out, look_left_only_mask)
            tgt_att_out = tgt_att_in + tgt_att_out
                        
            ffn_in = tgt_att_out
            #ffn_in has the data now 
            ffn_mid = self.ffn_norm[i](ffn_in)
            ffn_mid = self.dense_1[i](ffn_mid)
            
            ffn_out = self.dense_2[i](ffn_mid)

            ffn_out = ffn_out + ffn_in

            src_att_in = ffn_out
            #src_att_in has the data now

        return self.final_norm(src_att_in)
    
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
    