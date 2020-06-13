import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

from data_generator import DataGenerator
from networks import fully_connected, output_layer
from utils import log_results
from callbacks import DataGeneratorCallback, EvaluateCallback, LearningRateCallback


def train(hparams):

    inputs, network = fully_connected(hparams["input_size"], hparams["network_config"])
    outputs, loss_weights, losses = output_layer(network, hparams['output_config'])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="multi_detector")
    model.summary()

    opt = tf.keras.optimizer.'{}'.format(hparams['train_setup']['optimizer'])

    model.compile(optimizer=opt, loss=hparams['train_setup']['loss'])

    lr_callback = LearningRateCallback(hparams['train_setup']['learning_rate'])
    learn_rate_callback = LearningRateScheduler(lr_callback.calculate, verbose=1)

    dev_data_generator = DataGenerator(hparams, loss_weights, data_type='dev')
    dev_eval_callback = EvaluateCallback(dev_data_generator, "dev", hparams['dev_eval_period'])

    #for training in hparams["training_setup"]:
    #    train_params = copy.deepcopy(hparams)
    #    train_params["all_tasks"] = hparams["selected_tasks"]
    #    for k, v in training.items():
    #        train_params[k] = v

    #train_data_generator = DataGenerator(train_params, loss_weights)
    train_eval_callback = EvaluateCallback(
        train_data_generator, "train", hparams['train_setup']['train_eval_period'])

    model.fit(train_data_generator, verbose=1, shuffle=False,
              epochs=hparams["train_setup"]["epochs"],
              callbacks=[DataGeneratorCallback(train_data_generator),
                         learn_rate_callback, train_eval_callback, dev_eval_callback])

    test_data_generator = DataGenerator(hparams, loss_weights, data_type='test')
    evaluated_set = set()
    for k in ["geo", "sum"]:
        mean_type = k
        geo_epoch = dev_eval_callback.best_models["geo"]["epoch"]
        sum_epoch = dev_eval_callback.best_models["sum"]["epoch"]
        if geo_epoch == sum_epoch:
            mean_type = "geo-sum"

        if dev_eval_callback.best_models[k]["epoch"] not in evaluated_set:
            log_results(dev_eval_callback.best_models[k]["model"], hparams,
                        epochs=dev_eval_callback.best_models[k]["epoch"],
                        data_generator=test_data_generator, mean_type=mean_type)
            evaluated_set.add(dev_eval_callback.best_models[k]["epoch"])
