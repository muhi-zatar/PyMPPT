import tensorflow as tf

def fully_connected(input_size, network_config):
    inputs = tf.keras.Input(shape=(input_size,), dtype=tf.dtypes.float32)
    outputs = build_dense_layers(inputs, network_config, name="shared")
    return inputs, outputs


def build_dense_layers(inputs, network_config, name):
    hidden_nodes = network_config["hidden_nodes"]
    prev = inputs

    for i, size in enumerate(hidden_nodes):
        layer = tf.keras.layers.Dense(
            size, activation=network_config['activation'], name="{}_{}".format(name, i))(prev)
        prev = tf.keras.layers.Dropout(network_config['dropout'])(layer)

    return prev

def output_layer(inputs, output_config):
    output = tf.keras.layers.Dense(1, activation=output_config['activation'], name='output_layer')(inputs)
    return output
