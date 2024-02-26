import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense

# Class for one hot encoding and decoding strings
class CharacterTable(object):
	def __init__(self, chars):
		self.chars = sorted(set(chars))
		self.char_indices = dict((c,i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i,c) for i, c in enumerate(self.chars))

	def encode(self, string, num_rows):
		x = np.zeros((num_rows, len(self.chars)))
		for i, char in enumerate(string):
			if char == ' ': 
				continue
			x[i, self.char_indices[char]] = 1
		return x

	def decode(self, x, calc_argmax=True):
		if calc_argmax:
			x = x.argmax(axis=-1)
		return ''.join([self.indices_char[i] for i in x])

training_size = 50000
digits = 5
hidden_size = 128
batch_size = 128

# This is a pattern to match against input: X+Y
question_maxlen = digits + 1 + digits
chars = '0123456789+-'

ctable = CharacterTable(chars)

questions = []
answers = []
expected = []
seen = set()

while len(questions) < training_size:
	f = lambda: int(''.join(np.random.choice(list('0123456789'))
		for i in range(np.random.randint(1, digits+1))))
	a, b = f(), f()

	key = tuple(sorted([a, b]))
	if key in seen:
		continue

	seen.add(key)

	question = f'{a}-{b}'
	padding = ' ' * (question_maxlen - len(question))

	question = question + padding

	answer = str(a - b)
	padding = ' ' * (question_maxlen - len(answer))

	questions.append(question)
	answers.append(answer)
print("QnA Generated!")


# Encoding
x = np.zeros((len(questions), question_maxlen, len(chars)), dtype=bool)
y = np.zeros((len(questions), digits + 1, len(chars)), dtype=bool)
for i, question in enumerate(questions):
	x[i] = ctable.encode(question, question_maxlen)
for i, answer in enumerate(answers):
	y[i] = ctable.encode(answer, digits + 1)

# Make a random shuffling of the data
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Train test split (10% testing)
split = len(x) - len(x)//10

x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# Define the model
model = Sequential()
model.add(LSTM(hidden_size, input_shape=(question_maxlen, len(chars))))
model.add(RepeatVector(digits+1))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])
model.summary()

for iteration in range(1, 200):
	print(f"Iteration {iteration}")
	print('#'*25)
	model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=1,
			validation_data=(x_test, y_test))

	for i in range(10):
		ind = np.random.randint(0, len(x_test))
		rowx, rowy = x_test[np.array([ind])], y_test[np.array([ind])]
  
		# preds = model.predict_classes(rowx, verbose=0)
		preds = model.predict(rowx, verbose=0)
		preds = np.argmax(preds, axis=-1)
  
		question = ctable.decode(rowx[0])
		correct = ctable.decode(rowy[0])
		guess = ctable.decode(preds[0], calc_argmax=False)
		print("Q", question, end=' ')
		print("T", correct, end=' ')
		if correct == guess:
			print("T", end=' ')
		else:
			print("F", end=' ')
		print(guess)
