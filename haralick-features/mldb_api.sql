CREATE EXTENSION plpython3u;

DROP TABLE train_table;
DROP TABLE test_table;
DROP TABLE val_table;
DROP TABLE models_table;
DROP TABLE datasets;

CREATE TABLE IF NOT EXISTS datasets
(
    dataset_id serial primary key,
    dataset_name text
);

CREATE TABLE IF NOT EXISTS train_table
(
    sample_id serial primary key,
    dataset_id integer not null REFERENCES datasets ON DELETE CASCADE,
    x_train bytea not null,
    y_train integer not null
);

CREATE TABLE IF NOT EXISTS test_table
(
    sample_id serial primary key,
    dataset_id integer not null REFERENCES datasets ON DELETE CASCADE,
    x_test bytea not null,
    y_test integer not null
);

CREATE TABLE IF NOT EXISTS val_table
(
    sample_id serial primary key,
    dataset_id integer not null REFERENCES datasets ON DELETE CASCADE,
    x_val bytea not null,
    y_val integer not null
);

CREATE TABLE IF NOT EXISTS models_table
(
    model_id serial primary key,
    model_name text,
    dataset_id integer REFERENCES datasets ON DELETE RESTRICT,
    optimizer text,
    model_config jsonb,
    model_weights jsonb
);

SELECT * FROM models_table;
INSERT INTO models_table VALUES (2, 'conv2d-2', 1, 'adam');
DELETE FROM models_table WHERE model_id = 2;

CREATE OR REPLACE FUNCTION python_path()
RETURNS text
LANGUAGE 'plpython3u'
AS $BODY$
    import sys
    plpy.notice(f'pl/python3 Path: {sys.path[0]}')
    for path in sys.path[1:]:
        plpy.notice(path)
    return sys.path
$BODY$;

SELECT python_path();

CREATE OR REPLACE FUNCTION python_packages()
RETURNS text
LANGUAGE 'plpython3u'
AS $BODY$
    import pkg_resources
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        ["%s==%s" % (i.key, i.version) for i in installed_packages]
    )
    for package in installed_packages_list:
        plpy.notice(package)
    return installed_packages_list
$BODY$;

SELECT python_packages();

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

CREATE OR REPLACE FUNCTION load_dataset(dataset_name text, is_val_table boolean)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    # check dataset_name not in database
    dataset_ids = plpy.execute(f"select dataset_id from datasets where dataset_name = '{dataset_name}'")

    if dataset_ids.nrows() > 0:
        plpy.info(f"Dataset {dataset_name} already exists in database!")
        return f"Dataset {dataset_name} already exists in database!"

    # insert new dataset name
    plan = plpy.prepare("insert into datasets(dataset_name) values ($1)", ["text"])
    plpy.execute(plan, [dataset_name])

    # get new dataset_id
    dataset_id = plpy.execute(f"select dataset_id from datasets where dataset_name = '{dataset_name}'")[0]['dataset_id']

    for i in range(len(x_train)):
        plan = plpy.prepare("insert into train_table(dataset_id, x_train, y_train) values ($1, $2, $3)", ["int", "bytea", "int"])
        plpy.execute(plan, [dataset_id, x_train[i].tobytes(), y_train[i]])
        if i % (int(len(x_train) / 100)) == 0:
            plpy.notice(f"loaded {i / (int(len(x_train) / 100))}% train data")
    plpy.notice('---TRAIN-DATA-LOADED---')

    for i in range(len(x_test)):
        plan = plpy.prepare("insert into test_table(dataset_id, x_test, y_test) values ($1, $2, $3)", ["int", "bytea", "int"])
        plpy.execute(plan, [dataset_id, x_test[i].tobytes(), y_test[i]])
        if i % (int(len(x_test) / 100)) == 0:
            plpy.notice(f"loaded {i / (int(len(x_test) / 100))}% test data")
    plpy.notice('---TEST-DATA-LOADED---')

    if is_val_table:
        plpy.notice(f'val_table is enabled!')

        # select validation data
        x_val = x_test
        y_val = y_test

        for i in range(len(x_val)):
            plan = plpy.prepare("insert into val_table(dataset_id, x_val, y_val) values ($1, $2, $3)", ["int", "bytea", "int"])
            plpy.execute(plan, [dataset_id, x_val[i].tobytes(), y_val[i]])
            if i % (int(len(x_val) / 100)) == 0:
                plpy.notice(f"loaded {i / (int(len(x_val) / 100))}% val data")
        plpy.notice('---VAL-DATA-LOADED---')

    return f"Successful dataset {dataset_name} load!"
$BODY$;

SELECT * from datasets ORDER BY dataset_id;
SELECT * FROM train_table ORDER BY sample_id;
SELECT * FROM test_table ORDER BY sample_id;
SELECT * FROM val_table ORDER BY sample_id;

TRUNCATE TABLE datasets CASCADE;
TRUNCATE TABLE train_table;
TRUNCATE TABLE test_table;

SELECT load_dataset('mnist-3', false);

SELECT sample_id, dataset_name, x_train, y_train FROM train_table
JOIN datasets d on train_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_test, y_test FROM test_table
JOIN datasets d on test_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_val, y_val FROM val_table
JOIN datasets d on val_table.dataset_id = d.dataset_id
ORDER BY sample_id;

CREATE OR REPLACE FUNCTION show_sample(
    sample_table text,
    sample_id integer)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import numpy as np

    if sample_table in ["train", "test", "val"]:
        sample = plpy.execute(f"select x_{sample_table} from {sample_table}_table where sample_id = {sample_id}")

        if sample.nrows() == 0:
            plpy.info(f"Index {sample_id} out of data table \"{sample_table}\".")
            return f"Index {sample_id} out of data table \"{sample_table}\"."

        bytes_img = sample[0][f'x_{sample_table}']
        array_img = np.ndarray(shape=(28, 28), dtype=np.float64, buffer=bytes_img)

        for line in array_img:
            line_str = ''
            for num in line:
                if num != 0:
                    line_str += '* '
                else:
                    line_str += '. '
            plpy.notice(line_str)

        # get dataset name
        dataset_name = plpy.execute(
            f'select dataset_name from {sample_table}_table '
            f'join datasets d on {sample_table}_table.dataset_id = d.dataset_id '
            f'where sample_id = {sample_id}'
        )[0]['dataset_name']

        plpy.info(f"sample_id: {sample_id} dataset_name: {dataset_name}")
    else:
        plpy.info(f"Data table \"{sample_table}\" does not exist!")
        return f"Data table \"{sample_table}\" does not exist!"
    return "Successful sample show!"
$BODY$;

SELECT show_sample('train', 1);
SELECT show_sample('test', 1);
SELECT show_sample('train', 60001);
SELECT show_sample('val', 9000);

CREATE OR REPLACE FUNCTION define_and_save_model(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import numpy as np
    import tensorflow as tf
    from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    from datetime import datetime
    from tensorflow.python.keras.callbacks import LambdaCallback

    def get_list_from_sql(name, table):
        data_list = []
        sql_data = plpy.execute(f"select {name} from {table} order by id")
        for sample in sql_data:
            if name not in ['y_train', 'y_test']:
                array_img = np.ndarray(shape=(28, 28), dtype=np.float64, buffer=sample[f'{name}'])
                data_list.append(array_img)
            else:
                data_list.append(list(sample.values()))
        plpy.info(f"{name} loaded!")
        return np.squeeze(np.asarray(data_list, dtype='float'))

    x_train = get_list_from_sql('x_train', 'train_table')
    y_train = get_list_from_sql('y_train', 'train_table')
    x_test = get_list_from_sql('x_test', 'test_table')
    y_test = get_list_from_sql('y_test', 'test_table')

    model = keras.models.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
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
        f"insert into models_table (name, optimizer, model_config, model_weights)"
        f"values ('{model_name}', '{optimizer}', '{json_config}', '{json_weights}')"
    )

    return 'All is OK!'
$BODY$;

TRUNCATE TABLE models_table;
SELECT * FROM models_table;
SELECT define_and_save_model('conv2d-2');

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
        sql_data = plpy.execute(f"select {name} from {table} order by id")
        for sample in sql_data:
            if name not in ['y_train', 'y_test']:
                array_img = np.ndarray(shape=(28, 28), dtype=np.float64, buffer=sample[f'{name}'])
                data_list.append(array_img)
            else:
                data_list.append(list(sample.values()))
        plpy.info(f"{name} loaded!")
        return np.squeeze(np.asarray(data_list, dtype='float'))

    models_names = plpy.execute(f"select name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['name'])

    if model_name not in existing_names:
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

SELECT * FROM models_table;
SELECT load_and_test_model('conv2d-2');

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

    if model_name not in existing_names:
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
        array_img = np.ndarray(shape=(28, 28), dtype=np.float64, buffer=line['x_test'])
        sample_list.append(array_img)
    sample_list = np.squeeze(np.asarray(sample_list, dtype='float'))
    sample_list = sample_list.reshape(1, 28, 28)
    plpy.execute(f"select show_sample('test', {sample_id})")

    predict_value = new_model.predict(sample_list)
    count = 0
    for shell in predict_value:
        for value in shell:
            plpy.notice(f'{count} - {100 * value.astype(float):.5f}%')
            count += 1
    digit = np.argmax(predict_value)
    plpy.notice(f"digit = {digit}")

    return 'All is OK!'
$BODY$;

SELECT * FROM models_table;
SELECT test_random_sample('conv2d-2');

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

    if model_name not in existing_names:
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

    img = load_img('D:\\generatedfiles\\sample_image.png', grayscale=True, target_size=(28, 28))
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
    count = 0
    for shell in predict_value:
        for value in shell:
            plpy.notice(f'{count} - {100 * value.astype(float):.5f}%')
            count += 1
    digit = np.argmax(predict_value)
    plpy.notice(f"digit = {digit}")

    return 'All is OK!'
$BODY$;

SELECT * FROM models_table;
SELECT test_handwritten_sample('conv2d-2');
