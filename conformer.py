import tensorflow as tf
from tensorflow.keras import layers
from keras_tuner import HyperModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model


class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate):
        super(FeedForwardModule, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Define the components of the FeedForward module
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(d_model, activation=tf.nn.swish)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Apply layer normalization
        x = self.layer_norm(inputs)

        # First linear layer followed by swish activation and dropout
        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        # Second linear layer followed by dropout
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        return x


class RelativePositionMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(RelativePositionMultiHeadAttention, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)  # Q = K = V in self-attention
        attn_output = self.dropout(attn_output, training=training)
        out = self.layer_norm(inputs + attn_output)
        return out


class ConvolutionalModule1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dropout_rate):
        super(ConvolutionalModule1D, self).__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)

        # Pointwise convolution to change the number of filters
        self.pointwise_conv1 = tf.keras.layers.Conv1D(filters=2 * filters, kernel_size=kernel_size, padding='same')

        # Depthwise convolution applies separately to each feature map
        self.depthwise_conv = tf.keras.layers.SeparableConv1D(filters=2 * filters, kernel_size=kernel_size,
                                                              padding='same', depth_multiplier=1)

        self.batch_norm = tf.keras.layers.BatchNormalization()

        # Pointwise convolution to mix features after depthwise conv
        self.pointwise_conv2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.layer_norm(inputs)
        x = self.pointwise_conv1(x)

        # GLU activation
        channels = tf.split(x, num_or_size_splits=2, axis=-1)
        x = channels[0] * tf.sigmoid(channels[1])

        x = self.depthwise_conv(x)
        x = self.batch_norm(x, training=training)
        x = tf.nn.swish(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x, training=training)

        return x


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, kernel_size, dropout_rate):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout_rate)
        self.mhsa = RelativePositionMultiHeadAttention(d_model, num_heads, dropout_rate)
        self.conv = ConvolutionalModule1D(d_model, kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(d_model, dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # FFN module part 1
        ffn_out = self.ffn1(inputs, training=training)
        ffn_out_scaled = 0.5 * ffn_out
        a = inputs + ffn_out_scaled

        # Multi-Head Self-Attention (MHSA)
        mhsa_out = self.mhsa(a, training=training)
        b = a + mhsa_out

        # Convolution module
        conv_out = self.conv(b, training=training)
        c = b + conv_out

        # FFN module part 2 and final LayerNorm
        ffn_out_2 = self.ffn2(c, training=training)
        ffn_out_2_scaled = 0.5 * ffn_out_2
        y = self.layer_norm(c + ffn_out_2_scaled)

        return y


def build_model(input_shape, num_labels, d_model, num_heads, kernel_size, dropout_rate, subsample_stride):
    inputs = tf.keras.Input(shape=input_shape)

    # Example of Convolutional Subsampling
    x = tf.keras.layers.Conv2D(d_model, kernel_size=(4, 4), strides=subsample_stride, activation='relu',
                               padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # Linear layer (Dense layer applied on each time step)
    x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)  # Flatten features except batch and time dimension
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))(x)

    # Dropout
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = ConformerBlock(d_model, num_heads, kernel_size, dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)  # Use GlobalAveragePooling2D
    outputs = tf.keras.layers.Dense(num_labels, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


#build a hypermodel of a conformer model to be able to perform hyperparameter tuning
class ConformerHyperModel(HyperModel):
    def __init__(self, input_shape, num_labels):
        self.input_shape = input_shape
        self.num_labels = num_labels

    def build(self, hp):

        inputs = tf.keras.Input(shape=self.input_shape)

        # Hyperparameters
        d_model = hp.Int('d_model', min_value=128, max_value=256, step=32)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        kernel_size = hp.Int('kernel_size', min_value=3, max_value=8, step=1)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        # number of conformer blocks (max 2 otherwise model gets to big)
        num_blocks = hp.Int('num_blocks', min_value=1, max_value=2, step=1)

        #layeer before thee conformer block
        x = tf.keras.layers.Conv2D(filters=d_model, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Stack Conformer blocks according to the num_blocks hyperparameter
        for _ in range(num_blocks):
            x = ConformerBlock(d_model, num_heads, kernel_size, dropout_rate)(x)

        #pooling and softmax for multilabel classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_labels, activation='softmax')(x)

        #create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
        return model









