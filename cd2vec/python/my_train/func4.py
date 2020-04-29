def check_your_answer():
	from tensorflow.keras.applications import ResNet50
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2Dnum_classes = 2
	resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'my_new_model = Sequential()
	my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
	my_new_model.add(Dense(num_classes, activation='softmax'))my_new_model.layers[0].trainable = Falsestep_1.check()
