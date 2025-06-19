import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras import Sequential
import streamlit as st


# Load IMDb dataset
print("Loading dataset...")
(train_data, test_data), ds_info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Constants
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# Prepare dataset
train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Vectorization
print("Adapting TextVectorization layer...")
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

train_text = train_data.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Build model
print("Building model...")
model = Sequential([
    vectorize_layer,
    Embedding(VOCAB_SIZE, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
print("Training model...")
model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# Evaluate
print("Evaluating model...")
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.2f}")
model.save("saved_model")

# Prediction helper
def predict_sentiment(text: str):
    prediction = model.predict(tf.constant([text]))  # Proper input format
    sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"
    confidence = round(prediction[0][0]*100, 2)
    print(f"Review: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence}%)\n")



# Test predictions
predict_sentiment("The movie was absolutely fantastic and I loved it!")
predict_sentiment("It was boring, too long, and poorly acted.")
predict_sentiment("The movie was kind of good, I didn't like the storyline but I loved the characters and the plot and the songs")
