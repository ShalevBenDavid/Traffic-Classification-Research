import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers

labels = np.load("/home/shalev/Public/processed_labels.npy", allow_pickle=True)
sequences_padded = np.load("/home/shalev/Public/processed_sequences_padded.npy", allow_pickle=True)

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(sequences_padded, encoded_labels, test_size=0.2, random_state=42)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, n_classes, dropout=0,
                mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


input_shape = x_train.shape[1:]
n_classes = len(np.unique(encoded_labels))

model = build_model(
    input_shape,
    head_size=8,
    num_heads=1,
    ff_dim=2,
    num_transformer_blocks=2,
    mlp_units=[8],
    n_classes=n_classes,
    dropout=0.25,
    mlp_dropout=0.4
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)

model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)
