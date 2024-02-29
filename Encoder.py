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