import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

class LiteOCR:
	def __init__(self):
		self.vocab = ['character_01_ka','character_02_kha','character_03_ga','character_04_gha','character_05_kna','character_06_cha','character_07_chha','character_08_ja','character_09_jha','character_10_yna','character_11_taamatar','character_12_thaa','character_13_daa','character_14_dhaa','character_15_adna','character_16_tabala','character_17_tha','character_18_da','character_19_dha','character_20_na','character_21_pa','character_22_pha','character_23_ba','character_24_bha','character_25_ma','character_26_yaw','character_27_ra','character_28_la','character_29_waw','character_30_motosaw','character_31_petchiryakha','character_32_patalosaw','character_33_ha','character_34_chhya','character_35_tra','character_36_gya','digit_0','digit_1','digit_2','digit_3','digit_4','digit_5','digit_6','digit_7','digit_8','digit_9']
		self.img_rows = 32
		self.img_cols = 32

		self.CNN = LiteCNN()
		self.CNN.load_weights()

	def predict(self, image):
		X = np.reshape(image,(-1, 32, 32, 1))
		X = X.astype("float32")

		predicted_i = self.CNN.predict(X)
		# print('predicted_i',predicted_i[0].tolist())
		p_i = predicted_i[0].tolist()
		predicted_index = (p_i).index(max(p_i))
		print(p_i[predicted_index])
		# if(p_i[predicted_index] > 0.9):
		return self.vocab[predicted_index]
		# else:
		# 	return "unclear"

class LiteCNN:
	def __init__(self):
		num_classes = 46 #number of classes
		img_width = 32 
		img_height = 32
		img_depth = 1

		self.model = Sequential()
		self.model.add(Conv2D(50, 4, input_shape=(img_height, img_width, img_depth), activation='relu')) # input layer
		self.model.add(MaxPooling2D(pool_size=2))
		self.model.add(Conv2D(100, 4, activation='relu'))
		self.model.add(MaxPooling2D(pool_size=2))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(200, activation='relu'))
		self.model.add(Dense(num_classes, activation='softmax')) # output layer
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	def load_weights(self):
		self.model.load_weights('../devanagari_character_recognition_notebook/basic_cnn_10_epochs23.h5')
		
	def predict(self, X):
		XX = np.reshape(X,(-1, 32, 32, 1))
		return self.model.predict(XX,batch_size=1)
