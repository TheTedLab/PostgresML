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
    optimizer text,
    model_config jsonb,
    model_weights jsonb
);

CREATE OR REPLACE FUNCTION tf_version()
RETURNS text
LANGUAGE 'plpython3u'
AS $BODY$
    import tensorflow as tf
    plpy.notice(tf.config.list_physical_devices('GPU'))
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        plpy.notice(details.get('device_name', 'Unknown GPU'))
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
    import numpy as np

    if sample_table in ["train", "test"]:
        sample = plpy.execute(f"select x_{sample_table} from {sample_table}_table where id = {sample_id}")

        if sample.nrows() == 0:
            return f"Index {sample_id} out of data table \"{sample_table}\"."

        plpy.info(f"sample_id: {sample_id}")

        sample_list = np.squeeze(np.asarray([list(line.values()) for line in sample], dtype='float'))

        for line in sample_list:
            line_str = ''
            for num in line:
                if num != 0:
                    line_str += '* '
                else:
                    line_str += '. '
            plpy.info(line_str)
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
    import time

    start_time = time.time()
    new_train = []
    train_data = plpy.execute('select x_test from test_table')
    for line in train_data:
        new_train.append(list(line.values()))
    new_train = np.squeeze(np.asarray(new_train, dtype='float'))
    plpy.info(f"--- {(time.time() - start_time)} seconds ---")

    start_time = time.time()
    second_data = plpy.execute('select x_test from test_table')
    second_train = np.squeeze(np.asarray([list(line.values()) for line in second_data], dtype='float'))
    plpy.info(f"--- {(time.time() - start_time)} seconds ---")
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
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    optimizer = 'adam'

    model.compile(optimizer=optimizer,
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
                        epochs=6,
                        batch_size=128,
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
                    f"insert into models_table (name, optimizer, model_config, model_weights)"
                    f"values ('{model_name}', '{optimizer}', '{json_config}', '{json_weights}')"
    )

    return 'All is OK!'

$BODY$;

BEGIN;
SELECT * FROM models_table;
SELECT define_and_save_model('dense-128-5');
ROLLBACK;

CREATE OR REPLACE FUNCTION load_and_test_model(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import numpy as np
    import tensorflow as tf

    def get_list_from_sql(name, table):
        data_list = []
        sql_data = plpy.execute(f"select {name} from {table}")
        for line in sql_data:
            data_list.append(list(line.values()))
        plpy.info(f"{name} loaded!")
        return np.squeeze(np.asarray(data_list, dtype='float'))

    models_names = plpy.execute(f"select name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['name'])

    if not model_name in existing_names:
        return f"Model with name '{model_name} does not exist in the database!'"

    x_test = get_list_from_sql('x_test', 'test_table')
    y_test = get_list_from_sql('y_test', 'test_table')

    model_config = plpy.execute(f"select model_config from models_table where name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    score = new_model.evaluate(x_test, y_test, verbose=0)
    plpy.notice(f"Test loss: {score[0]}")
    plpy.notice(f"Test accuracy: {score[1]}")

    return 'All is OK!'

$BODY$;

BEGIN;
SELECT * FROM models_table;
SELECT load_and_test_model('dense-128-5');
ROLLBACK;

CREATE OR REPLACE FUNCTION test_random_sample(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import random
    import numpy as np
    import tensorflow as tf

    models_names = plpy.execute(f"select name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['name'])

    if not model_name in existing_names:
        return f"Model with name '{model_name} does not exist in the database!'"

    model_config = plpy.execute(f"select model_config from models_table where name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    sample_id = random.randint(0, 9999)
    sample = plpy.execute(f"select x_test from test_table where id = {sample_id}")
    sample_list = []
    for line in sample:
        sample_list.append(list(line.values()))
    sample_list = np.squeeze(np.asarray(sample_list, dtype='float'))
    sample_list = sample_list.reshape(1, 28, 28)
    plpy.execute(f"select show_sample('test', {sample_id})")

    predict_value = new_model.predict(sample_list)
    digit = np.argmax(predict_value)
    plpy.notice(f"digit = {digit}")

    return 'All is OK!'
$BODY$;

BEGIN;
SELECT * FROM models_table;
SELECT test_random_sample('dense-128-5');
ROLLBACK;

CREATE OR REPLACE FUNCTION test_handwritten_sample(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from PIL import Image, ImageChops

    models_names = plpy.execute(f"select name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['name'])

    if not model_name in existing_names:
        return f"Model with name '{model_name} does not exist in the database!'"

    model_config = plpy.execute(f"select model_config from models_table where name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    img = load_img('C:\\generatedfiles\\sample_image.png', grayscale=True, target_size=(28, 28))
    img = ImageChops.invert(img)
    sample = img_to_array(img)
    sample = sample / 255

    for line in sample:
        line_str = ''
        for num in line:
            if num != 0:
                line_str += '* '
            else:
                line_str += '. '
        plpy.info(line_str)

    sample = sample.reshape(1, 28, 28)
    sample = sample.astype('float')

    predict_value = new_model.predict(sample)
    digit = np.argmax(predict_value)
    plpy.notice(f"digit = {digit}")

    return 'All is OK!'
$BODY$;

SELECT test_handwritten_sample('dense-128-5');
