CREATE EXTENSION plpython3u;

CREATE TABLE IF NOT EXISTS train_table
(
    id serial primary key,
    x_train numeric[28][28] not null,
    y_train integer not null
);

CREATE TABLE IF NOT EXISTS test_table
(
    id serial primary key,
    x_test numeric[28][28] not null,
    y_train integer not null
);

CREATE TABLE IF NOT EXISTS models_table
(
    id serial primary key,
    name text,
    model_config jsonb,
    model_weights jsonb
);

CREATE OR REPLACE FUNCTION tf_version()
RETURNS text
LANGUAGE 'plpython3u'
AS $BODY$
    import tensorflow as tf
    return tf.__version__
$BODY$;

SELECT tf_version();

CREATE OR REPLACE FUNCTION load_mnist()
    RETURNS void
    LANGUAGE 'plpython3u'
AS $BODY$
    import tensorflow as tf


    def list_to_pgsql(id, x_lists, y_num):
        array_str = ''
        lists_count = len(x_lists)
        for x_list in x_lists:
            lists_count -= 1
            array_str += '{'
            x_count = len(x_list)
            for x in x_list:
                x_count -= 1
                if x_count > 0:
                    array_str += str(x) + ', '
                else:
                    array_str += str(x)
            if lists_count > 0:
                array_str += '}, '
            else:
                array_str += '}'
        return str(id) + ', \'{' + array_str + '}\', ' + str(y_num)


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    for i in range(len(x_train)):
        val_array = list_to_pgsql(i, x_train[i], y_train[i])
        plpy.execute(f"insert into train_table values ({val_array})")
        if i % 600 == 0:
            plpy.notice(f"loaded {i / 600}% train data")
    plpy.notice('---TRAIN-DATA-LOADED---')

    for i in range(len(x_test)):
        val_array = list_to_pgsql(i, x_test[i], y_test[i])
        plpy.execute(f"insert into test_table values ({val_array})")
        if i % 100 == 0:
            plpy.notice(f"loaded {i / 100}% test data")
    plpy.notice('---TEST-DATA-LOADED---')
$BODY$;

BEGIN;
SELECT load_mnist();
SELECT * from train_table order by id;
SELECT * from test_table order by id;
ROLLBACK;

TRUNCATE TABLE train_table;
TRUNCATE TABLE test_table;

CREATE OR REPLACE FUNCTION show_sample(
    sample_table text,
    sample_id integer)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    from math import ceil


    if sample_table in ["train", "test"]:
        sample = plpy.execute(f"select * from {sample_table}_table where id = {sample_id}")
        if sample.nrows() == 0:
            return f"Index {sample_id} out of data table \"{sample_table}\"."

        mass = sample[0][f"x_{sample_table}"]
        for line in mass:
            line_str = ''
            for num in line:
                if num != 0:
                    line_str += '* '
                else:
                    line_str += '. '
            plpy.notice(line_str)
    else:
        return f"Data table \"{sample_table}\" does not exist!"
    return "Successful sample show!"
$BODY$;

BEGIN;
SELECT show_sample('train', 0);
SELECT show_sample('test', 0);
SELECT show_sample('train_data', 0);
SELECT show_sample('test', 10000);
ROLLBACK;

CREATE OR REPLACE FUNCTION test_loading()
    RETURNS void
    LANGUAGE 'plpython3u'
AS $BODY$
    import numpy as np

    new_train = []
    train_data = plpy.execute('select x_train from train_table where id <= 10')
    for line in train_data:
        new_train.append(list(line.values()))
    plpy.info(np.squeeze(np.asarray(new_train, dtype='float')))


$BODY$;

BEGIN;
SELECT test_loading();
ROLLBACK;

CREATE OR REPLACE FUNCTION define_and_save_model(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import numpy as np
    import tensorflow as tf
    from keras.layers import Dense, Flatten
    from datetime import datetime
    from tensorflow.python.keras.callbacks import LambdaCallback

    def get_list_from_sql(name, table):
        data_list = []
        sql_data = plpy.execute(f"select {name} from {table}")
        for line in sql_data:
            data_list.append(list(line.values()))
        plpy.info(f"{name} loaded!")
        return np.squeeze(np.asarray(data_list, dtype='float'))

    x_train = get_list_from_sql('x_train', 'train_table')
    y_train = get_list_from_sql('y_train', 'train_table')
    x_test = get_list_from_sql('x_test', 'test_table')
    y_test = get_list_from_sql('y_test', 'test_table')

    model = keras.models.Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    plpy.notice('Model architecture:\n{}'.format('\n'.join(summary)))

    logger = LambdaCallback(
        on_epoch_end=lambda epoch,
        logs: plpy.notice(f"epoch: {epoch}, accuracy {logs['accuracy']:.4f}, loss: {logs['loss']:.4f}")
    )

    plpy.notice('create logger')

    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(x_test, y_test),
                        verbose=False,
                        callbacks=[logger])

    plpy.notice('model fit complete')

    json_config = model.to_json()
    model_weights = model.get_weights()

    for i in range(len(model_weights)):
        model_weights[i] = model_weights[i].tolist()

    json_weights = json.dumps(model_weights)

    plpy.notice('json conversions complete')

    plpy.execute(
                    f"insert into models_table (name, model_config, model_weights)"
                    f"values ('{model_name}', '{json_config}', '{json_weights}')"
    )

    return 'All is OK!'

$BODY$;

BEGIN;
SELECT * FROM models_table;
SELECT define_and_save_model('dense-64-x2');
ROLLBACK;
