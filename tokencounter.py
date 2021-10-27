import datahandler
import midisave

data = datahandler.NumpyFileShuffle(prm['N_TIMESTEPS'], filename_t=f"C:\\datasets\\midi\\lakh_training_tokens_v4.npy", filename_v=f"C:\\datasets\\midi\\validation_tokens_{midisave.version}.npy", vocab_size=prm['VOCAB_SIZE'])

megainit = data.get_mega_init()