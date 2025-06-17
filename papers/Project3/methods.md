
- Use Latin Hyper Cube for sampling cosmological parameters and redshift. 
- Generate data in a step (use same k sampling as given by classy_sz)
- in another step train the 1D CNN neural net. A good architecture is:
    inputs = keras.Input(shape=(input_dim,))
    # Project input to output_dim for convolution
    x = layers.Dense(output_dim, activation='relu')(inputs)
    x = layers.Reshape((output_dim, 1))(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(output_dim)(x)
- in another step train the FCNN (with 4 hidden layers with 512 nodes each and swish activation function)
- in another step compare the performance and show some diagnostic plots and true vs emulated pk at various redshifts. 
