import numpy as np
import tensorflow as tf

import counting

def reduce(matrix, axis):
    mean = tf.math.reduce_mean(matrix, axis=axis, keepdims=True)
    stddev = tf.math.reduce_std(matrix, axis=axis, keepdims=True)
    return (matrix - mean)/(stddev + 1e-2)

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, n_heads, normalize_columns, use_causal_mask):
        super(MultiHeadAttention, self).__init__()
        self.model_size = model_size
        self.head_size = model_size // n_heads
        self.n_heads = n_heads

        self.w_q = tf.keras.layers.Dense(model_size)
        self.w_kv = tf.keras.layers.Dense(model_size*2)

        self.wo = tf.keras.layers.Dense(model_size)
        self.normalizer = 1.0/tf.math.sqrt(tf.dtypes.cast(self.head_size, tf.float32))
        self.normalize_columns = normalize_columns
        
        self.use_causal_mask = use_causal_mask
        
    def call(self, query, value, mask=None):
        heads = []
        out_kv = self.w_kv(value)
        out_q = self.w_q(query)
        
        if self.normalize_columns:
            out_kv = reduce(out_kv,axis=-2)
            out_q = reduce(out_q,axis=-2)

        out_kv = tf.split(out_kv, self.n_heads*2, axis=-1)
        out_q = tf.split(out_q, self.n_heads, axis=-1)
        
        for i in range(self.n_heads):
            wqd = out_q[i]
            wkd = out_kv[i]
            wvd = out_kv[i+self.n_heads]
            
            score = tf.matmul(wqd, wkd, transpose_b=True)*self.normalizer
            if mask is not None:
                score += mask
            #if self.use_causal_mask:
                #mask_size = score.shape[-1]
                #look_left_only_mask = (tf.linalg.band_part(tf.ones((mask_size,mask_size)), -1, 0)-1)*90000000.0
                #score = score+look_left_only_mask

            alignment = tf.nn.softmax(score, axis=-1)
            head = tf.matmul(alignment, wvd)
            heads.append(head)

        heads = tf.concat(heads, axis=-1)
        heads = tf.reshape(heads, [query.shape[0],-1,self.model_size])
        heads = self.wo(heads)
        return heads

#no positional encoding, no classification layers, no frills.
#basic transformer is all you need :D
#
#this version of transformer attends to the source sequence *first*
#only then to the target sequence
class Transformer(tf.keras.Model):
    def __init__(self, model_size, num_layers, heads, use_causal_mask, use_counter, normalize_columns, abstraction_size=None):
        super(Transformer, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.heads = heads

        self.src_att = [MultiHeadAttention(model_size, heads, use_causal_mask=use_causal_mask, normalize_columns=normalize_columns) for _ in range(num_layers)]
        self.src_att_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        self.tgt_att = [MultiHeadAttention(model_size, heads, use_causal_mask=use_causal_mask, normalize_columns=normalize_columns) for _ in range(num_layers)]
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
            
        
        
    def call(self, target_sequence, source_sequence=None, src_as_list=False):
        if self.use_causal_mask:
            mask_size = target_sequence.shape[1]
            look_left_only_mask = tf.linalg.band_part(tf.ones((mask_size,mask_size)), -1, 0)
            look_left_only_mask = (look_left_only_mask-1)*90000000.0
        else:
            look_left_only_mask = None
            
        src_att_in = target_sequence
        for i in range(self.num_layers):
            #src_att_in has the data now
            
            if self.use_counter:
                src_att_in = self.counting[i](src_att_in)
                src_att_in = self.counting_norm[i](src_att_in)
        
            if source_sequence is not None:
                if src_as_list:
                    #src_att_out = []
                    #for seq_item in source_sequence:
                    #    src_att_out.append(self.src_att[i](src_att_in, seq_item))
                    #src_att_out = tf.add_n(src_att_out)
                    
                    processed_seq = tf.concat(source_sequence,axis=0)
                    src_att_out = self.src_att_norm[i](src_att_in)
                    src_att_out = self.src_att[i](src_att_out, processed_seq, None)
                    
                else:
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
    
