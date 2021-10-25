from keras.layers import Dense, LSTM, Input, Dropout, Permute, multiply
from keras.models import Model

def Att_LSTM():
    input = Input(shape=(40, 2048), name='input')
    fc_0 = Dense(4096, activation='relu', name='fc_0')(input)
    attention = attention_block(fc_0, 40)
    lstm = LSTM(512, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)(attention)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(lstm)
    drop_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(drop_1)
    drop_2 = Dropout(0.5)(fc_2)
    out = Dense(12, activation='softmax', name='classification')(drop_2)
    model = Model(input, out, name='att-lstm')

    return model

def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')

    return output_attention_mul

# from attention import BahdanauAttention

# n_timesteps_in = 40
# n_features = 4096
# latentSpaceDimension = 100
# batch_size = 1

# # The first part is encoder
# encoder_inputs = Input(shape=(n_timesteps_in, n_features), name='encoder_inputs')
# encoder_lstm = LSTM(latentSpaceDimension, return_sequences=True, return_state=True, name='encoder_lstm')
# encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)

# # initial context vector is the states of the encoder
# encoder_states = [encoder_state_h, encoder_state_c]

# # Set up the attention layer
# attention= BahdanauAttention(latentSpaceDimension)

# # Set up the decoder layers
# decoder_inputs = Input(shape=(1, (n_features+latentSpaceDimension)),name='decoder_inputs')
# decoder_lstm = LSTM(latentSpaceDimension,  return_state=True, name='decoder_lstm')
# decoder_dense = Dense(n_features, activation='softmax',  name='decoder_dense')

# all_outputs = []

# # 1 initial decoder's input data
# # Prepare initial decoder input data that just contains the start character 
# # Note that we made it a constant one-hot-encoded in the model
# # that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
# # one-hot encoded zero(0) is the start symbol
# inputs = np.zeros((batch_size, 1, n_features))
# inputs[:, 0, 0] = 1 


# # 2 initial decoder's state
# # encoder's last hidden state + last cell state
# decoder_outputs = encoder_state_h
# states = encoder_states

# # decoder will only process one time step at a time.
# for _ in range(n_timesteps_in):

#     # 3 pay attention
#     # create the context vector by applying attention to 
#     # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
#     context_vector, attention_weights=attention(decoder_outputs, encoder_outputs)

#     context_vector = K.expand_dims(context_vector, 1)

#     # 4. concatenate the input + context vectore to find the next decoder's input
#     inputs = K.concatenate([context_vector, inputs], axis=-1)

#     # 5. passing the concatenated vector to the LSTM
#     # Run the decoder on one timestep with attended input and previous states
#     decoder_outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
#     #decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
  
#     outputs = decoder_dense(decoder_outputs)
#     # 6. Use the last hidden state for prediction the output
#     # save the current prediction
#     # we will concatenate all predictions later
#     outputs = K.expand_dims(outputs, 1)
#     all_outputs.append(outputs)
#     # 7. Reinject the output (prediction) as inputs for the next loop iteration
#     # as well as update the states
#     inputs = outputs
#     states = [state_h, state_c]

# # 8. After running Decoder for max time steps
# # we had created a predition list for the output sequence
# # convert the list to output array by Concatenating all predictions 
# # such as [batch_size, timesteps, features]
# decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

# # 9. Define and compile model 
# model_encoder_decoder_Bahdanau_Attention = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
# model_encoder_decoder_Bahdanau_Attention.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])