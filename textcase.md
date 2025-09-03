code:

import numpy as np from tensorflow.keras.preprocessing.text import Tokenizer from tensorflow.keras.utils import pad_sequences, to_categorical from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Embedding, SimpleRNN, Dense data = "Deep learning is amazing. Deep learning builds intelligent systems." tokenizer = Tokenizer() tokenizer.fit_on_texts([data]) sequences = [] words = data.split() for i in range(1, len(words)): seq = words[:i+1] sequences.append(' '.join(seq)) encoded = tokenizer.texts_to_sequences(sequences) max_len = max([len(x) for x in encoded])

X = np.array([x[:-1] for x in pad_sequences(encoded, maxlen=max_len)]) y = to_categorical([x[-1] for x in pad_sequences(encoded, maxlen=max_len)], num_classes=len(tokenizer.word_index)+1) model = Sequential([ Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=max_len-1), SimpleRNN(50), Dense(len(tokenizer.word_index)+1, activation='softmax') ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) model.fit(X, y, epochs=200, verbose=0) def predict_next_word(text, true_word=None): """Predict the next word for a given input text and check if correct""" seq = tokenizer.texts_to_sequences([text])[0] padded = pad_sequences([seq], maxlen=max_len-1) pred = model.predict(padded, verbose=0) predicted_id = np.argmax(pred)

predicted_word = [w for w, idx in tokenizer.word_index.items() if idx == predicted_id][0] correct = "N" if true_word and predicted_word.lower() == true_word.lower(): correct = "Y"

return predicted_word, correct test_cases = [ ("Deep learning is", "amazing."), ("Deep learning builds intelligent", "systems."), ("Intelligent systems can learn", "something") print(f"{'Input Text':40s} {'Predicted Word':15s} {'Correct (Y/N)'}") for text, true_word in test_cases: pred_word, result = predict_next_word(text, true_word) print(f"{text:40s} {pred_word:15s} {result}") 


screenshot:<img width="1131" height="110" alt="ex4 testcase" src="https://github.com/user-attachments/assets/a1e26499-0743-433d-9c35-24b6284f89eb" />
