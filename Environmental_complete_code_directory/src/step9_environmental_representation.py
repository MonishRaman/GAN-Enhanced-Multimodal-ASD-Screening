import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. Load processed environmental data
# ============================================================

X_train = pd.read_csv(
    "X_train_selected20.csv"
).values

X_test = pd.read_csv(
    "X_test_selected20.csv"
).values

y_train = pd.read_csv(
    "y_train_selected20.csv"
).iloc[:, 0].values

y_test = pd.read_csv(
    "y_test_selected20.csv"
).iloc[:, 0].values


print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 3. Class weights
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# 4. Build Environmental Representation Network
# ============================================================

input_dim = X_train.shape[1]

inputs = Input(
    shape=(input_dim,),
    name="environmental_input"
)

x = Dense(
    128,
    activation="relu",
    name="dense_128"
)(inputs)

x = Dropout(
    0.30
)(x)

x = Dense(
    64,
    activation="relu",
    name="dense_64"
)(x)

x = Dropout(
    0.20
)(x)


# ============================================================
# 5. 128-dimensional environmental embedding
# ============================================================

embedding = Dense(
    128,
    activation="linear",
    name="environmental_embedding"
)(x)


# ============================================================
# 6. ASD prediction head
# ============================================================

prediction = Dense(
    1,
    activation="sigmoid",
    name="asd_output"
)(embedding)


# ============================================================
# 7. Complete model
# ============================================================

model = Model(
    inputs=inputs,
    outputs=prediction
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ============================================================
# 8. Model summary
# ============================================================

print("\nModel architecture:")
model.summary()


# ============================================================
# 9. Callbacks
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)


# ============================================================
# 10. Train
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 11. Test prediction
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_prediction = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# 12. Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\n==========================================")
print("ENVIRONMENTAL MODEL TEST RESULTS")
print("==========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_prediction,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 13. Create embedding model
# ============================================================

embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer(
        "environmental_embedding"
    ).output
)


# ============================================================
# 14. Generate 128-D environmental embeddings
# ============================================================

train_embeddings = embedding_model.predict(
    X_train,
    verbose=0
)

test_embeddings = embedding_model.predict(
    X_test,
    verbose=0
)


print("\nEmbedding shapes:")
print(
    "Train:",
    train_embeddings.shape
)

print(
    "Test :",
    test_embeddings.shape
)


# ============================================================
# 15. Save embeddings
# ============================================================

pd.DataFrame(
    train_embeddings
).to_csv(
    "environmental_embedding_train.csv",
    index=False
)

pd.DataFrame(
    test_embeddings
).to_csv(
    "environmental_embedding_test.csv",
    index=False
)


# ============================================================
# 16. Save model
# ============================================================

model.save(
    "environmental_representation_model.keras"
)

print("\nSaved:")
print("environmental_embedding_train.csv")
print("environmental_embedding_test.csv")
print("environmental_representation_model.keras")