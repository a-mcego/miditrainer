import tensorflow as tf

import posenc

class CountingAbstraction(tf.keras.Model):
    def __init__(self, model_size):
        super(CountingAbstraction, self).__init__()
        
        self.counter_expander = tf.keras.layers.Dense(model_size,activation=tf.nn.softplus)
        self.counter_merger = tf.keras.layers.Dense(model_size,use_bias=False)
        
    def call(self, data):
        #cosine similarity
        counter = tf.math.l2_normalize(data, axis=-1)
        counter = tf.matmul(counter, counter, transpose_b=True)
        #relu to filter out bads (this could be improved upon)
        counter = tf.nn.relu(counter)

        #do the fixed-v abstraction
        fixed_v = posenc.get_posenc(data.shape[-2],data.shape[-1])
        v_output = tf.matmul(counter,fixed_v)

        #do the count itself
        counter = tf.reduce_sum(counter,axis=-1,keepdims=True)
        
        #combine them
        counter = tf.concat([counter,v_output],axis=-1)
        
        #do postprocessing of the count.
        counter = self.counter_expander(counter)
        #merge the data here.
        data = self.counter_merger(tf.concat([data,counter],axis=-1))
        return data

class Counting(tf.keras.Model):
    def __init__(self, model_size):
        super(Counting, self).__init__()
        
        self.counter_expander = tf.keras.layers.Dense(model_size,activation=tf.nn.softplus)
        self.counter_merger = tf.keras.layers.Dense(model_size,use_bias=False)
        
    def call(self, data):
        #cosine similarity
        counter = tf.math.l2_normalize(data, axis=-1)
        counter = tf.matmul(counter, counter, transpose_b=True)
        #relu to filter out bads (this could be improved upon)
        counter = tf.nn.relu(counter)
        #do the count itself
        counter = tf.reduce_sum(counter,axis=-1,keepdims=True)
        #do postprocessing of the count.
        counter = self.counter_expander(counter)
        #merge the data here.
        data = self.counter_merger(tf.concat([data,counter],axis=-1))
        return data

class Abstraction(tf.keras.Model):
    def __init__(self, model_size, abstraction_size, use_causal_mask):
        super(Abstraction, self).__init__()
        self.use_causal_mask = use_causal_mask
        self.abstraction_size = abstraction_size
        
        self.abstract_vec_maker = tf.keras.layers.Dense(model_size,use_bias=False)
        self.merger = tf.keras.layers.Dense(model_size,use_bias=False)
        
    def call(self, data):
        #cosine similarity
        counter = data
        counter = tf.math.l2_normalize(counter, axis=-1)
        counter = tf.matmul(counter, counter, transpose_b=True)
        if self.use_causal_mask:
            mask_size = counter.shape[-1]
            look_left_only_mask = tf.linalg.band_part(tf.ones((mask_size,mask_size)), -1, 0)
            counter = tf.multiply(counter,look_left_only_mask)

        
        added_size = (self.abstraction_size-counter.shape[2]%self.abstraction_size)%self.abstraction_size
        
        added_zeros = tf.zeros(shape=[counter.shape[0],counter.shape[1],added_size], dtype=counter.dtype)
        
        counter = tf.concat([counter,added_zeros],axis=-1)
        counter = tf.reshape(counter,[counter.shape[0],counter.shape[1],-1,self.abstraction_size])
        counter = tf.reduce_mean(counter,axis=-2)
        
        abstract_vectors = self.abstract_vec_maker(counter)
        data = self.merger(tf.concat([data,abstract_vectors],axis=-1))
        return data

class AbstractionPosEnc(tf.keras.Model):
    def __init__(self, model_size, abstraction_size, use_causal_mask):
        super(AbstractionPosEnc, self).__init__()
        self.use_causal_mask = use_causal_mask
        self.abstraction_size = abstraction_size
        
        self.abstract_vec_maker = tf.keras.layers.Dense(model_size,use_bias=False)
        self.merger = tf.keras.layers.Dense(model_size,use_bias=False)
        
    def call(self, data):
        #cosine similarity
        counter = data
        #counter = tf.math.l2_normalize(counter, axis=-1)
        counter = tf.matmul(counter, counter, transpose_b=True)
        if self.use_causal_mask:
            mask_size = counter.shape[-1]
            look_left_only_mask = tf.linalg.band_part(tf.ones((mask_size,mask_size)), -1, 0)
            counter = tf.multiply(counter,look_left_only_mask)

        
        added_size = (self.abstraction_size-counter.shape[2]%self.abstraction_size)%self.abstraction_size
        
        added_zeros = tf.zeros(shape=[counter.shape[0],counter.shape[1],added_size], dtype=counter.dtype)
        
        counter = tf.concat([counter,added_zeros],axis=-1)
        counter = tf.reshape(counter,[counter.shape[0],counter.shape[1],-1,self.abstraction_size])
        counter = tf.reduce_mean(counter,axis=-2)
        
        abstract_vectors = self.abstract_vec_maker(counter)

        pes = posenc.get_standard_posenc(data.shape[-2],data.shape[-1])
        pes = tf.expand_dims(pes,axis=0)
        pes = tf.broadcast_to(pes,[data.shape[0],pes.shape[1],pes.shape[2]])
        data = self.merger(tf.concat([data,abstract_vectors,pes],axis=-1))
        return data

class AbstractionTimeChange(tf.keras.Model):
    def __init__(self, model_size, timesteps_out):
        super(Abstraction, self).__init__()
        
        self.abstract_vec_maker = tf.keras.layers.Dense(model_size,use_bias=False)
        self.merger = tf.keras.layers.Dense(model_size,use_bias=False)
        self.timestep_changer(timesteps_out)
        
    def call(self, data):
        #B batch, T1 timesteps_in, T2 timesteps_out, d model_size
    
    
        #cosine similarity
        # B x T1 x d
        counter = data 
        
        # B x T1 x d
        counter = tf.math.l2_normalize(counter, axis=-1)
        
        # B x T1 x T1
        counter = tf.matmul(counter, counter, transpose_b=True)
        
        # B x T1 x d
        abstract_vectors = self.abstract_vec_maker(counter)

        # T1 x d
        pes = posenc.get_posenc(data.shape[-2],data.shape[-1])
        
        # 1 x T1 x d
        pes = tf.expand_dims(pes,axis=0)
        
        # B x T1 x d
        pes = tf.broadcast_to(pes,[data.shape[0],pes.shape[1],pes.shape[2]])
        
        # B x T1 x (d + d + d)  ==  B x T1 x 3d
        concatted = tf.concat([data,abstract_vectors,pes],axis=-1)
        
        # B x 3d x T1
        concatted = tf.transpose(concatted, perm=[0,2,1])
        
        # B x 3d x T2
        concatted = self.timestep_changer(concatted)

        # B x T2 x 3d
        concatted = tf.transpose(concatted, perm=[0,2,1])
        
        # B x T2 x d
        data = self.merger(concatted)

        return data

