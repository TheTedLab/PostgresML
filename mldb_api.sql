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

SELECT tf_version();
SELECT show_sample('train', 81,
    'gray');
SELECT load_dataset('D:\\haralick-dataset',
    'haralick',true);
SELECT glcm_digitization('haralick', true);
SELECT noise_generation('haralick',true,
    0.032,10);
SELECT define_and_save_model('haralick',true,
    true,'wheat-network',
    'D:\\configs\\model_arch.yaml');
SELECT load_and_test_model('conv2d-12');
SELECT test_digital_sample('conv2d-10', 123);
SELECT test_original_image('conv2d-12', 1);

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
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    plpy.notice(tf.config.list_physical_devices('GPU'))
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        plpy.notice(details.get('device_name', 'Unknown GPU'))
    return tf.__version__
$BODY$;

SELECT tf_version();

CREATE OR REPLACE FUNCTION lib_versions_test()
RETURNS text
LANGUAGE 'plpython3u'
AS $BODY$
    import numpy
    import skimage
    import sklearn
    import scipy
    import tifffile

    versions = {
        'numpy': numpy.__version__,
        'skimage': skimage.__version__,
        'sklearn': sklearn.__version__,
        'scipy': scipy.__version__,
        'tifffile': tifffile.__version__
    }
    plpy.notice(f"numpy=={versions['numpy']}")
    plpy.notice(f"scikit-image=={versions['skimage']}")
    plpy.notice(f"scikit-learn=={versions['sklearn']}")
    plpy.notice(f"scipy=={versions['scipy']}")
    plpy.notice(f"tifffile=={versions['tifffile']}")

    return versions
$BODY$;

SELECT lib_versions_test();

CREATE OR REPLACE FUNCTION load_dataset(
    dataset_path text,
    dataset_name text,
    is_val_table boolean)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
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

    import os
    import skimage
    import pickle

    x_train, y_train = [], []
    train_dir = os.path.join(dataset_path, 'train_dir')
    for class_id in [name for name in os.listdir(train_dir)]:
        class_dir = os.path.join(train_dir, class_id)
        plpy.notice(class_dir)
        for sample in os.listdir(class_dir):
            sample_dir = os.path.join(class_dir, sample)
            image = skimage.io.imread(sample_dir)
            x_train.append(image)
            y_train.append(str(int(class_id) - 1))

    for i in range(len(x_train)):
        plan = plpy.prepare("insert into train_table(dataset_id, x_train, y_train) values ($1, $2, $3)", ["int", "bytea", "int"])
        plpy.execute(plan, [dataset_id, pickle.dumps(x_train[i]), y_train[i]])
        if len(x_train) > 100 and i % (int(len(x_train) / 100)) == 0:
            plpy.notice(f"loaded {i / (int(len(x_train) / 100))}% train data")
        elif len(x_train) < 100:
            plpy.notice(f"loaded {i / (len(x_train) / 100)}% train data")
    plpy.notice('---TRAIN-DATA-LOADED---')

    x_test, y_test = [], []
    test_dir = os.path.join(dataset_path, 'test_dir')
    for class_id in [name for name in os.listdir(test_dir)]:
        class_dir = os.path.join(test_dir, class_id)
        plpy.notice(class_dir)
        for sample in os.listdir(class_dir):
            sample_dir = os.path.join(class_dir, sample)
            image = skimage.io.imread(sample_dir)
            x_test.append(image)
            y_test.append(str(int(class_id) - 1))

    for i in range(len(x_test)):
        plan = plpy.prepare("insert into test_table(dataset_id, x_test, y_test) values ($1, $2, $3)", ["int", "bytea", "int"])
        plpy.execute(plan, [dataset_id, pickle.dumps(x_test[i]), y_test[i]])
        if len(x_test) > 100 and i % (int(len(x_test) / 100)) == 0:
            plpy.notice(f"loaded {i / (int(len(x_test) / 100))}% test data")
        elif len(x_test) < 100:
            plpy.notice(f"loaded {i / (len(x_test) / 100)}% test data")
    plpy.notice('---TEST-DATA-LOADED---')

    if is_val_table:
        plpy.notice(f'val_table is enabled!')

        # select validation data
        x_val, y_val = [], []
        val_dir = os.path.join(dataset_path, 'val_dir')
        for class_id in [name for name in os.listdir(val_dir)]:
            class_dir = os.path.join(val_dir, class_id)
            plpy.notice(class_dir)
            for sample in os.listdir(class_dir):
                sample_dir = os.path.join(class_dir, sample)
                image = skimage.io.imread(sample_dir)
                x_val.append(image)
                y_val.append(str(int(class_id) - 1))

        for i in range(len(x_val)):
            plan = plpy.prepare("insert into val_table(dataset_id, x_val, y_val) values ($1, $2, $3)", ["int", "bytea", "int"])
            plpy.execute(plan, [dataset_id, pickle.dumps(x_val[i]), y_val[i]])
            if len(x_val) > 100 and i % (int(len(x_val) / 100)) == 0:
                plpy.notice(f"loaded {i / (int(len(x_val) / 100))}% val data")
            elif len(x_val) < 100:
                plpy.notice(f"loaded {i / (len(x_val) / 100)}% val data")
        plpy.notice('---VAL-DATA-LOADED---')

    return f"Successful dataset {dataset_name} load!"
$BODY$;

SELECT * from datasets ORDER BY dataset_id;
SELECT * FROM train_table ORDER BY sample_id;
SELECT * FROM test_table ORDER BY sample_id;
SELECT * FROM val_table ORDER BY sample_id;

SELECT load_dataset(
    'D:\\haralick-dataset',
    'haralick',
    true
);

SELECT * from datasets ORDER BY dataset_id;

SELECT sample_id, dataset_name, x_train, y_train FROM train_table
JOIN datasets d on train_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_test, y_test FROM test_table
JOIN datasets d on test_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_val, y_val FROM val_table
JOIN datasets d on val_table.dataset_id = d.dataset_id
ORDER BY sample_id;

CREATE OR REPLACE FUNCTION load_mnist(
    dataset_name text,
    is_val_table boolean)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
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

    import tensorflow as tf
    import pickle

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    for i in range(len(x_train)):
        plan = plpy.prepare("insert into train_table(dataset_id, x_train, y_train) values ($1, $2, $3)", ["int", "bytea", "int"])
        plpy.execute(plan, [dataset_id, pickle.dumps(x_train[i]), y_train[i]])
        if i % (int(len(x_train) / 100)) == 0:
            plpy.notice(f"loaded {i / (int(len(x_train) / 100))}% train data")
    plpy.notice('---TRAIN-DATA-LOADED---')

    for i in range(len(x_test)):
        plan = plpy.prepare("insert into test_table(dataset_id, x_test, y_test) values ($1, $2, $3)", ["int", "bytea", "int"])
        plpy.execute(plan, [dataset_id, pickle.dumps(x_test[i]), y_test[i]])
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
            plpy.execute(plan, [dataset_id, pickle.dumps(x_val[i]), y_val[i]])
            if i % (int(len(x_val) / 100)) == 0:
                plpy.notice(f"loaded {i / (int(len(x_val) / 100))}% val data")
        plpy.notice('---VAL-DATA-LOADED---')

    return f"Successful dataset {dataset_name} load!"
$BODY$;

SELECT * from datasets ORDER BY dataset_id;

SELECT sample_id, dataset_name, x_train, y_train FROM train_table
JOIN datasets d on train_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_test, y_test FROM test_table
JOIN datasets d on test_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_val, y_val FROM val_table
JOIN datasets d on val_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT load_mnist('mnist', false);

CREATE OR REPLACE FUNCTION show_sample(
    sample_table text,
    sample_id integer,
    color_map text DEFAULT 'viridis')
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    if sample_table in ["train", "test", "val"]:
        sample = plpy.execute(f"select x_{sample_table} from {sample_table}_table where sample_id = {sample_id}")

        if sample.nrows() == 0:
            plpy.info(f"Index {sample_id} out of data table \"{sample_table}\".")
            return f"Index {sample_id} out of data table \"{sample_table}\"."

        bytes_img = sample[0][f'x_{sample_table}']
        array_img = pickle.loads(bytes_img)

        plt.clf()
        plt.imshow(array_img, cmap=color_map)
        plt.savefig(f'D:\\saved-images\\sample-{sample_table}-{sample_id}.png')
        plt.close()

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
    return f"Successful show! sample_id = {sample_id}, dataset: {dataset_name}"
$BODY$;

SELECT show_sample('train', 1);
SELECT show_sample('test', 1);
SELECT show_sample('val', 1);
SELECT show_sample('train', 81, 'gray');
SELECT show_sample('test', 17, 'gray');
SELECT show_sample('val', 25, 'gray');

CREATE OR REPLACE FUNCTION show_mnist(
    sample_table text,
    sample_id integer)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import pickle
    import numpy as np

    if sample_table in ["train", "test", "val"]:
        sample = plpy.execute(f"select x_{sample_table} from {sample_table}_table where sample_id = {sample_id}")

        if sample.nrows() == 0:
            plpy.info(f"Index {sample_id} out of data table \"{sample_table}\".")
            return f"Index {sample_id} out of data table \"{sample_table}\"."

        bytes_img = sample[0][f'x_{sample_table}']
        array_img = pickle.loads(bytes_img)

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
    return f"Successful show! sample_id = {sample_id}, dataset: {dataset_name}"
$BODY$;

SELECT show_mnist('train', 100);
SELECT show_mnist('test', 100);

CREATE OR REPLACE FUNCTION glcm_digitization(
    dataset_name text,
    is_val_table boolean)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import pickle
    import skimage
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.feature.texture import graycomatrix, graycoprops

    def calc_component_features(img_component):
        img_component = np.true_divide(img_component, 32)
        img_component = img_component.astype(int)
        glcm = graycomatrix(img_component, [1], [0], levels=8, symmetric=False,
                            normed=True)
        haralick_features = {
            'correlation': graycoprops(glcm, 'correlation')[0, 0],
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0]
        }
        return haralick_features

    # check dataset_name-digital not in database
    dataset_ids = plpy.execute(f"select dataset_id from datasets where dataset_name = '{dataset_name}-digital'")

    if dataset_ids.nrows() > 0:
        plpy.info(
            f'Dataset {dataset_name}-digital already exists in database. '
            f'You have either already digitized or created a dataset with a reserved name.'
        )
        return f'Dataset {dataset_name}-digital already exists in database. ' \
               f'You have either already digitized or created a dataset with a reserved name.'

    # insert new dataset name
    plan = plpy.prepare("insert into datasets(dataset_name) values ($1)", ["text"])
    plpy.execute(plan, [f'{dataset_name}-digital'])

    # get new dataset_id
    dataset_id = plpy.execute(f"select dataset_id from datasets where dataset_name = '{dataset_name}-digital'")[0]['dataset_id']

    tables_list = ['train', 'test']
    if is_val_table:
        tables_list.append('val')
    for table_name in tables_list:
        samples = plpy.execute(
            f'select dataset_name, x_{table_name}, y_{table_name} from {table_name}_table '
            f'join datasets d on {table_name}_table.dataset_id = d.dataset_id '
            f'where dataset_name = \'{dataset_name}\''
        )

        if samples.nrows() == 0:
            plpy.info(f'No samples in {table_name}_table for dataset with name \"{dataset_name}\".')
            return f'No samples in {table_name}_table for dataset with name \"{dataset_name}\".'

        img_RED_global = []
        img_GREEN_global = []
        img_BLUE_global = []
        for sample in samples:
            bytes_img = sample[f'x_{table_name}']
            array_img = pickle.loads(bytes_img)

            img_components = {}

            # RED component
            img_red = array_img[:, :, 0]
            img_RED_global = img_red
            img_components['R'] = calc_component_features(img_red)

            # GREEN component
            img_green = array_img[:, :, 2]
            img_GREEN_global = img_green
            img_components['G'] = calc_component_features(img_green)

            # BLUE component
            img_blue = array_img[:, :, 0]
            img_BLUE_global = img_blue
            img_components['B'] = calc_component_features(img_blue)

            # RED-GREEN component
            img_r_g = img_RED_global - img_GREEN_global
            img_components['RG'] = calc_component_features(img_r_g)

            # RED-BLUE component
            img_r_b = img_RED_global - img_BLUE_global
            img_components['RB'] = calc_component_features(img_r_b)

            # GREEN-BLUE component
            img_g_b = img_GREEN_global - img_BLUE_global
            img_components['GB'] = calc_component_features(img_g_b)

            # construct an image
            preprocessed_image = np.zeros([4, 6])
            comp_index = 0
            for component in img_components.values():
                feature_index = 0
                for key, val in component.items():
                    preprocessed_image[feature_index][comp_index] = val
                    feature_index += 1
                comp_index += 1

            # save image
            fig = plt.figure(frameon=False)
            fig.set_size_inches(0.06, 0.04)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(preprocessed_image, aspect='auto', cmap='Greys')
            image_path = 'D:\\saved-images\\digital_image.png'
            plt.savefig(image_path)
            plt.close(fig)

            # load digital image to database
            digital_image = skimage.io.imread(image_path, as_gray=True)
            y_lable = sample[f'y_{table_name}']

            if table_name == 'train':
                plan = plpy.prepare("insert into train_table(dataset_id, x_train, y_train) values ($1, $2, $3)", ["int", "bytea", "int"])
                plpy.execute(plan, [dataset_id, pickle.dumps(digital_image), y_lable])
            elif table_name == 'test':
                plan = plpy.prepare("insert into test_table(dataset_id, x_test, y_test) values ($1, $2, $3)", ["int", "bytea", "int"])
                plpy.execute(plan, [dataset_id, pickle.dumps(digital_image), y_lable])
            elif table_name == 'val':
                plan = plpy.prepare("insert into val_table(dataset_id, x_val, y_val) values ($1, $2, $3)", ["int", "bytea", "int"])
                plpy.execute(plan, [dataset_id, pickle.dumps(digital_image), y_lable])
    return "Successful glcm digitization!"
$BODY$;

SELECT glcm_digitization('haralick', true);

SELECT * from datasets ORDER BY dataset_id;

SELECT sample_id, dataset_name, x_train, y_train FROM train_table
JOIN datasets d on train_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_test, y_test FROM test_table
JOIN datasets d on test_table.dataset_id = d.dataset_id
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_val, y_val FROM val_table
JOIN datasets d on val_table.dataset_id = d.dataset_id
ORDER BY sample_id;

CREATE OR REPLACE FUNCTION noise_generation(
    dataset_name text,
    is_val_table boolean,
    standard_deviation double precision,
    noise_amount int)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import pickle
    import random
    import skimage
    import numpy as np
    import matplotlib.pyplot as plt

    def make_noise_and_insert(image, noise_range, diff, dataset_id, table_name, y_lable):
        for noises in range(noise_range):
            for i in range(len(image)):
                for j in range(len(image[i])):
                    image[i][j] += random.uniform(-standard_deviation, standard_deviation)

            # save image
            fig = plt.figure(frameon=False)
            fig.set_size_inches(0.06, 0.04)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(image, aspect='auto', cmap='gray')
            image_path = f'D:\\saved-images\\noise_image.png'
            plt.savefig(image_path)

            # load noise image to database
            noise_image = plt.imread(image_path)[:,:,:3]
            plt.close(fig)

            if table_name == 'train':
                plan = plpy.prepare("insert into train_table(dataset_id, x_train, y_train) values ($1, $2, $3)", ["int", "bytea", "int"])
                plpy.execute(plan, [dataset_id, pickle.dumps(noise_image), y_lable])
            elif table_name == 'test':
                plan = plpy.prepare("insert into test_table(dataset_id, x_test, y_test) values ($1, $2, $3)", ["int", "bytea", "int"])
                plpy.execute(plan, [dataset_id, pickle.dumps(noise_image), y_lable])
            elif table_name == 'val':
                plan = plpy.prepare("insert into val_table(dataset_id, x_val, y_val) values ($1, $2, $3)", ["int", "bytea", "int"])
                plpy.execute(plan, [dataset_id, pickle.dumps(noise_image), y_lable])


    def delete_dataset(dataset_id):
        plpy.execute(f'delete from datasets where dataset_id = {dataset_id}')

    # check dataset_name-digital exists in database
    digital_dataset = plpy.execute(f'select dataset_id from datasets where dataset_name = \'{dataset_name}-digital\'')

    if digital_dataset.nrows() == 0:
        plpy.info(
            f'Dataset {dataset_name}-digital does not exist in the database.'
        )
        return f'Dataset {dataset_name}-digital does not exist in the database.'

    # check dataset_name-noised not in database
    dataset_ids = plpy.execute(f"select dataset_id from datasets where dataset_name = '{dataset_name}-noised'")

    if dataset_ids.nrows() > 0:
        plpy.info(
            f'Dataset {dataset_name}-noised already exists in database. '
            f'You have either already generated noise or created a dataset with a reserved name.'
        )
        return f'Dataset {dataset_name}-noise already exists in database. ' \
               f'You have either already generated noise or created a dataset with a reserved name.'

    # insert new dataset name
    plan = plpy.prepare("insert into datasets(dataset_name) values ($1)", ["text"])
    plpy.execute(plan, [f'{dataset_name}-noised'])

    # get new dataset_id
    dataset_id = plpy.execute(f"select dataset_id from datasets where dataset_name = '{dataset_name}-noised'")[0]['dataset_id']

    # check that train_table is not empty
    check_samples = plpy.execute(
        f'select y_train from train_table '
        f'join datasets d on train_table.dataset_id = d.dataset_id '
        f'where dataset_name = \'{dataset_name}-digital\''
    )

    if check_samples.nrows() == 0:
        plpy.info(
            f'Dataset {dataset_name}-digital does not contain samples in train_table.'
        )
        delete_dataset(dataset_id)
        return f'Dataset {dataset_name}-digital does not contain samples in train_table.'

    # get min and max classes labels
    min_class = plpy.execute(
        f'select min(y_train) from train_table '
        f'join datasets d on train_table.dataset_id = d.dataset_id '
        f'where dataset_name = \'{dataset_name}-digital\''
    )[0]['min']
    max_class = plpy.execute(
        f'select max(y_train) from train_table '
        f'join datasets d on train_table.dataset_id = d.dataset_id '
        f'where dataset_name = \'{dataset_name}-digital\''
    )[0]['max']

    for class_id in range(min_class, max_class + 1):
        class_samples = []
        samples_train = plpy.execute(
            f'select dataset_name, x_train, y_train from train_table '
            f'join datasets d on train_table.dataset_id = d.dataset_id '
            f'where dataset_name = \'{dataset_name}-digital\' and y_train = {class_id}'
        )

        if samples_train.nrows() == 0:
            plpy.info(f'No samples in train_table for dataset with name \"{dataset_name}-digital\".')
            delete_dataset(dataset_id)
            return f'No samples in train_table for dataset with name \"{dataset_name}-digital\".'

        class_samples += samples_train

        samples_test = plpy.execute(
            f'select dataset_name, x_test, y_test from test_table '
            f'join datasets d on test_table.dataset_id = d.dataset_id '
            f'where dataset_name = \'{dataset_name}-digital\' and y_test = {class_id}'
        )

        if samples_test.nrows() == 0:
            plpy.info(f'No samples in test_table for dataset with name \"{dataset_name}-digital\".')
            delete_dataset(dataset_id)
            return f'No samples in test_table for dataset with name \"{dataset_name}-digital\".'

        class_samples += samples_test

        if is_val_table:
            samples_val = plpy.execute(
                f'select dataset_name, x_val, y_val from val_table '
                f'join datasets d on val_table.dataset_id = d.dataset_id '
                f'where dataset_name = \'{dataset_name}-digital\' and y_val = {class_id}'
            )

            if samples_val.nrows() == 0:
                plpy.info(f'No samples in val_table for dataset with name \"{dataset_name}-digital\".')
                delete_dataset(dataset_id)
                return f'No samples in val_table for dataset with name \"{dataset_name}-digital\".'

            class_samples += samples_val

        total_class_img = [[0., 0., 0., 0., 0., 0.] for _ in range(4)]
        for sample in class_samples:
            bytes_img = None
            table_name = ''
            if sample.get('x_train') is not None:
                bytes_img = sample.get('x_train')
                table_name = 'x_train'
            elif sample.get('x_test') is not None:
                bytes_img = sample.get('x_test')
                table_name = 'x_test'
            elif sample.get('x_val') is not None:
                bytes_img = sample.get('x_val')
                table_name = 'x_val'
            array_img = pickle.loads(bytes_img)

            # get average digital image
            for i in range(len(array_img)):
                for j in range(len(array_img[0])):
                    total_class_img[i][j] += array_img[i][j]

        average_img = np.true_divide(total_class_img, 15)

        make_noise_and_insert(
            average_img,
            len(samples_train) * noise_amount,
            0,
            dataset_id,
            'train',
            class_id
        )
        make_noise_and_insert(
            average_img,
            len(samples_test) * noise_amount,
            len(samples_train) * noise_amount,
            dataset_id,
            'test',
            class_id
        )
        if is_val_table:
            make_noise_and_insert(
                average_img,
                len(samples_val) * noise_amount,
                len(samples_train) * noise_amount + len(samples_test) * noise_amount,
                dataset_id,
                'val',
                class_id
            )
    return "Successful noise generation!"
$BODY$;

SELECT noise_generation(
    'haralick',
    true,
    0.032,
    10
);

SELECT * from datasets ORDER BY dataset_id;

SELECT sample_id, dataset_name, x_train, y_train FROM train_table
JOIN datasets d on train_table.dataset_id = d.dataset_id
WHERE dataset_name = 'haralick-noised'
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_test, y_test FROM test_table
JOIN datasets d on test_table.dataset_id = d.dataset_id
WHERE dataset_name = 'haralick-noised'
ORDER BY sample_id;

SELECT sample_id, dataset_name, x_val, y_val FROM val_table
JOIN datasets d on val_table.dataset_id = d.dataset_id
WHERE dataset_name = 'haralick-noised'
ORDER BY sample_id;

SELECT show_sample('train', 161, 'gray');
SELECT show_sample('test', 33, 'gray');
SELECT show_sample('val', 49, 'gray');

CREATE OR REPLACE FUNCTION define_and_save_model(
    dataset_name text,
    is_val_table boolean,
    is_noised_data boolean,
    model_name text,
    config_path text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import pickle
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from datetime import datetime
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
    from tensorflow.python.keras.callbacks import LambdaCallback

    dataset_postfix = None
    if is_noised_data:
        dataset_postfix = '-noised'
    else:
        dataset_postfix = ''

    # check dataset_name(-noised) in database
    dataset_ids = plpy.execute(f'select dataset_id from datasets where dataset_name = \'{dataset_name}{dataset_postfix}\'')

    if dataset_ids.nrows() == 0:
        plpy.info(
            f'Dataset {dataset_name} does not exists in database. '
        )
        return f'Dataset {dataset_name} does not exists in database. '

    dataset_id = dataset_ids[0]['dataset_id']

    x_train, y_train, x_test, y_test, x_val, y_val  = [], [], [], [], None, None
    tables_list = ['train', 'test']
    if is_val_table:
        tables_list.append('val')
    for table_name in tables_list:
        samples = plpy.execute(
            f'select x_{table_name}, y_{table_name} from {table_name}_table '
            f'join datasets d on {table_name}_table.dataset_id = d.dataset_id '
            f'where dataset_name = \'{dataset_name}{dataset_postfix}\''
        )

        if samples.nrows() == 0:
            plpy.info(f'No samples in {table_name}_table for dataset with name \"{dataset_name}\".')
            return f'No samples in {table_name}_table for dataset with name \"{dataset_name}\".'

        if is_val_table:
            x_val, y_val = [], []
        for sample in samples:
            bytes_img = sample[f'x_{table_name}']
            x_data = pickle.loads(bytes_img)
            y_data = sample[f'y_{table_name}']

            if table_name == 'train':
                x_train.append(x_data)
                y_train.append(y_data)
            elif table_name == 'test':
                x_test.append(x_data)
                y_test.append(y_data)
            elif table_name == 'val':
                x_val.append(x_data)
                y_val.append(y_data)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if is_val_table:
        x_val = np.array(x_val)
        y_val = np.array(y_val)
    else:
        x_val = np.array(x_test)
        y_val = np.array(y_test)

    # get amount of classes
    classes_num = plpy.execute(
        f'select count(distinct y_train) from train_table '
        f'join datasets d on train_table.dataset_id = d.dataset_id '
        f'where dataset_name = \'{dataset_name}{dataset_postfix}\';'
    )[0]['count']

    # input shape dimension for image data in format: (width, height, channels)
    input_shape = (4, 6, 3)
    # amount of epochs
    epochs = 200
    # mini-batch size
    batch_size = 10

    # amount of samples for training
    nb_train_samples = int(len(x_train) / classes_num)
    # amount of samples for validation
    nb_validation_samples = int(len(x_val) / classes_num)

    datagen = ImageDataGenerator()

    train_generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size
    )

    val_generator = datagen.flow(
        x_val,
        y_val,
        batch_size=batch_size,
    )

    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    optimizer = 'adam'

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    plpy.notice('Model architecture:\n{}'.format('\n'.join(summary)))

    logger = LambdaCallback(
        on_epoch_end=lambda epoch,
        logs: plpy.notice(
            f"epoch: {epoch + 1}, acc: {logs['accuracy']:.4f}, loss: {logs['loss']:.4f} "
            f"val_acc: {logs['val_accuracy']:.4f}, val_loss: {logs['val_loss']:.4f}"
        )
    )

    plpy.notice('create logger')

    history = model.fit(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        validation_data=val_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=False,
                        callbacks=[logger])

    plpy.notice('model fit complete')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs,
             acc, 'bo', label='Training acc')
    plt.plot(epochs,
             val_acc, color='orange', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    accuracy_path = f'D:\\saved-images\\graphs\\{model_name}-train-and-val-acc.png'
    plt.savefig(accuracy_path, bbox_inches='tight')

    plt.figure()
    plt.plot(epochs,
             loss, 'bo', label='Smoothed training loss')
    plt.plot(epochs,
             val_loss, color='orange', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    loss_path = f'D:\\saved-images\\graphs\\{model_name}-train-and-val-loss.png'
    plt.savefig(loss_path, bbox_inches='tight')

    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    plt.figure()
    plt.plot(epochs,
             smooth_curve(acc), 'bo', label='Training acc')
    plt.plot(epochs,
             smooth_curve(val_acc), color='orange', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    smooth_accuracy_path = f'D:\\saved-images\\graphs\\{model_name}-smooth-train-and-val-acc.png'
    plt.savefig(smooth_accuracy_path, bbox_inches='tight')

    plt.figure()
    plt.plot(epochs,
             smooth_curve(loss), 'bo', label='Training loss')
    plt.plot(epochs,
             smooth_curve(val_loss), color='orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    smooth_loss_path = f'D:\\saved-images\\graphs\\{model_name}-smooth-train-and-val-loss.png'
    plt.savefig(smooth_loss_path, bbox_inches='tight')
    plt.clf()
    plt.close()

    plpy.notice('graphs draw complete')

    json_config = model.to_json()
    model_weights = model.get_weights()

    for i in range(len(model_weights)):
        model_weights[i] = model_weights[i].tolist()

    json_weights = json.dumps(model_weights)

    plpy.notice('json conversions complete')

    plpy.execute(
        f"insert into models_table (model_name, dataset_id, optimizer, model_config, model_weights)"
        f"values ('{model_name}', {dataset_id}, '{optimizer}', '{json_config}', '{json_weights}')"
    )

    return 'All is OK!'
$BODY$;

TRUNCATE TABLE models_table;
SELECT * FROM models_table;
SELECT * FROM datasets;

SELECT define_and_save_model(
    'haralick',
    true,
    true,
    'wheat-network',
    'D:\\configs\\model_arch.yaml'
);

CREATE OR REPLACE FUNCTION load_and_test_model(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import pickle
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    models_names = plpy.execute(f"select model_name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['model_name'])

    if model_name not in existing_names:
        plpy.info(f'Model with name \'{model_name}\' does not exist in the database!')
        return f'Model with name \'{model_name}\' does not exist in the database!'

    model_dataset_id = plpy.execute(f'select dataset_id from models_table where model_name = \'{model_name}\'')[0]['dataset_id']

    x_test, y_test  = [], []
    samples = plpy.execute(
        f'select x_test, y_test from test_table '
        f'where dataset_id = {model_dataset_id}'
    )

    if samples.nrows() == 0:
        dataset_name = plpy.execute(
            f'select dataset_name from datasets where dataset_id = {model_dataset_id}'
        )[0]['dataset_name']
        plpy.info(f'No samples in test_table for dataset with name \"{dataset_name}\".')
        return f'No samples in test_table for dataset with name \"{dataset_name}\".'

    for sample in samples:
        bytes_img = sample[f'x_test']
        x_data = pickle.loads(bytes_img)
        y_data = sample[f'y_test']

        x_test.append(x_data)
        y_test.append(y_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model_config = plpy.execute(f"select model_config from models_table where model_name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where model_name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where model_name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    test_loss, test_acc = new_model.evaluate(x_test, y_test)
    plpy.notice(f"Test loss: {test_loss}")
    plpy.notice(f"Test accuracy: {test_acc}")

    y_pred_raw = new_model.predict(x_test)

    y_pred = np.argmax(y_pred_raw, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(0, 8, 1), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(1, 9, 1))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Greens', ax=ax)
    ax.set_xlabel('Предсказанные метки', fontsize=16)
    ax.set_ylabel('Истинные метки', fontsize=16)
    ax.set_title('Матрица ошибок', fontsize=20)
    ax.set_xticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
    ax.set_yticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
    plt.tight_layout()
    confusion_matrix_norm_path = f'D:\\saved-images\\graphs\\{model_name}-confusion-matrix-normalized.png'
    plt.savefig(confusion_matrix_norm_path, bbox_inches='tight')
    plt.clf()

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(0, 8, 1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(1, 9, 1))
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Greens', ax=ax)
    ax.set_xlabel('Предсказанные метки', fontsize=16)
    ax.set_ylabel('Истинные метки', fontsize=16)
    ax.set_title('Матрица ошибок', fontsize=20)
    ax.set_xticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
    ax.set_yticks(np.arange(0, 8, 1), labels=np.arange(1, 9, 1), fontsize=16)
    plt.tight_layout()
    confusion_matrix_path = f'D:\\saved-images\\graphs\\{model_name}-confusion-matrix.png'
    plt.savefig(confusion_matrix_path, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

    return 'All is OK!'
$BODY$;

SELECT * FROM models_table;
SELECT load_and_test_model('conv2d-12');

CREATE OR REPLACE FUNCTION test_random_sample(model_name text)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import random
    import pickle
    import numpy as np
    import tensorflow as tf

    models_names = plpy.execute(f"select model_name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['model_name'])

    if model_name not in existing_names:
        plpy.info(f'Model with name \'{model_name}\' does not exist in the database!')
        return f'Model with name \'{model_name}\' does not exist in the database!'

    model_config = plpy.execute(f"select model_config from models_table where model_name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where model_name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where model_name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model_dataset_id = plpy.execute(f'select dataset_id from models_table where model_name = \'{model_name}\'')[0]['dataset_id']

    x_len = plpy.execute(f'select count(sample_id) from test_table where dataset_id = {model_dataset_id}')[0]['count']

    sample_id = random.randint(0, x_len)
    plpy.notice(f'random sample_id: {sample_id}')

    dataset_offset = plpy.execute(f'select min(sample_id) from test_table where dataset_id = {model_dataset_id}')[0]['min'] - 1
    plpy.notice(f'dataset_offset: {dataset_offset}')
    plpy.notice(f'dataset_offset + sample_id: {dataset_offset + sample_id}')

    samples = plpy.execute(
        f'select x_test, y_test from test_table '
        f'where dataset_id = {model_dataset_id} and sample_id = {dataset_offset + sample_id}'
    )

    if samples.nrows() == 0:
        dataset_name = plpy.execute(
            f'select dataset_name from datasets where dataset_id = {model_dataset_id}'
        )[0]['dataset_name']
        plpy.info(f'No samples in test_table for dataset with name \"{dataset_name}\".')
        return f'No samples in test_table for dataset with name \"{dataset_name}\".'

    x_test, y_test = [], []
    for sample in samples:
        bytes_img = sample[f'x_test']
        x_data = pickle.loads(bytes_img)
        y_data = sample[f'y_test']

        x_test.append(x_data)
        y_test.append(y_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)[0]

    plpy.execute(f"select show_sample('test', {dataset_offset + sample_id}, 'gray')")

    predict_value = new_model.predict(x_test)
    count = 1
    for shell in predict_value:
        for value in shell:
            plpy.notice(f'{count} - {100 * value.astype(float):.5f}%')
            count += 1
    predict_result = np.argmax(predict_value)
    plpy.notice(f"predict = {predict_result + 1}")
    plpy.notice(f"true = {y_test + 1}")

    return f"sample_id = {sample_id}, predict = {predict_result + 1}, true = {y_test + 1}"
$BODY$;

SELECT min(sample_id) FROM test_table WHERE dataset_id = 3;
SELECT * FROM models_table;
SELECT test_random_sample('conv2d-12');

CREATE OR REPLACE FUNCTION test_digital_sample(
    model_name text,
    test_sample_id integer)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import pickle
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    models_names = plpy.execute(f"select model_name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['model_name'])

    if model_name not in existing_names:
        plpy.info(f'Model with name \'{model_name}\' does not exist in the database!')
        return f'Model with name \'{model_name}\' does not exist in the database!'

    model_config = plpy.execute(f"select model_config from models_table where model_name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where model_name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where model_name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    plpy.notice(f'test_sample_id: {test_sample_id}')

    samples = plpy.execute(
        f'select x_test, y_test from test_table '
        f'where sample_id = {test_sample_id}'
    )

    if samples.nrows() == 0:
        plpy.info(f'No sample in test_table for sample_id = {test_sample_id}.')
        return f'No sample in test_table for sample_id = {test_sample_id}.'

    x_test, y_test = [], []
    for sample in samples:
        bytes_img = sample[f'x_test']
        x_data = pickle.loads(bytes_img)
        y_data = sample[f'y_test']

        x_test.append(x_data)
        y_test.append(y_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)[0]

    plpy.execute(f"select show_sample('test', {test_sample_id}, 'gray')")

    predict_value = new_model.predict(x_test)
    count = 1
    for shell in predict_value:
        for value in shell:
            plpy.notice(f'{count} - {100 * value.astype(float):.5f}%')
            count += 1
    predict_result = np.argmax(predict_value)
    plpy.notice(f"predict = {predict_result + 1}")
    plpy.notice(f"true = {y_test + 1}")

    return f"sample_id = {test_sample_id}, predict = {predict_result + 1}, true = {y_test + 1}"
$BODY$;

SELECT * FROM test_table;
SELECT * FROM models_table;
SELECT test_digital_sample(
    'conv2d-10', 123
);

CREATE OR REPLACE FUNCTION test_original_image(
    model_name text,
    test_sample_id integer)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$
    import json
    import keras
    import pickle
    import skimage
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from skimage.feature.texture import graycomatrix, graycoprops

    def calc_component_features(img_component):
        img_component = np.true_divide(img_component, 32)
        img_component = img_component.astype(int)
        glcm = graycomatrix(img_component, [1], [0], levels=8, symmetric=False,
                            normed=True)
        haralick_features = {
            'correlation': graycoprops(glcm, 'correlation')[0, 0],
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0]
        }
        return haralick_features

    models_names = plpy.execute(f"select model_name from models_table")
    existing_names = []
    for sql_name in models_names:
        existing_names.append(sql_name['model_name'])

    if model_name not in existing_names:
        plpy.info(f'Model with name \'{model_name}\' does not exist in the database!')
        return f'Model with name \'{model_name}\' does not exist in the database!'

    model_config = plpy.execute(f"select model_config from models_table where model_name = '{model_name}'")
    new_model = keras.models.model_from_json(model_config[0]['model_config'])

    model_weights = plpy.execute(f"select model_weights from models_table where model_name = '{model_name}'")

    json_weights = json.loads(model_weights[0]['model_weights'])

    for i in range(len(json_weights)):
        json_weights[i] = np.array(json_weights[i])

    new_model.set_weights(json_weights)

    new_optimizer = plpy.execute(f"select optimizer from models_table where model_name = '{model_name}'")

    new_model.compile(optimizer=new_optimizer[0]['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    plpy.notice(f'test_sample_id: {test_sample_id}')

    sample = plpy.execute(
        f'select x_test, y_test from test_table '
        f'where sample_id = {test_sample_id}'
    )

    if sample.nrows() == 0:
        plpy.info(f'No sample in test_table for sample_id = {test_sample_id}.')
        return f'No sample in test_table for sample_id = {test_sample_id}.'

    plpy.execute(f"select show_sample('test', {test_sample_id})")

    img_RED_global = []
    img_GREEN_global = []
    img_BLUE_global = []
    bytes_img = sample[0][f'x_test']
    array_img = pickle.loads(bytes_img)

    img_components = {}

    # RED component
    img_red = array_img[:, :, 0]
    img_RED_global = img_red
    img_components['R'] = calc_component_features(img_red)

    # GREEN component
    img_green = array_img[:, :, 2]
    img_GREEN_global = img_green
    img_components['G'] = calc_component_features(img_green)

    # BLUE component
    img_blue = array_img[:, :, 0]
    img_BLUE_global = img_blue
    img_components['B'] = calc_component_features(img_blue)

    # RED-GREEN component
    img_r_g = img_RED_global - img_GREEN_global
    img_components['RG'] = calc_component_features(img_r_g)

    # RED-BLUE component
    img_r_b = img_RED_global - img_BLUE_global
    img_components['RB'] = calc_component_features(img_r_b)

    # GREEN-BLUE component
    img_g_b = img_GREEN_global - img_BLUE_global
    img_components['GB'] = calc_component_features(img_g_b)

    # construct an image
    preprocessed_image = np.zeros([4, 6])
    comp_index = 0
    for component in img_components.values():
        feature_index = 0
        for key, val in component.items():
            preprocessed_image[feature_index][comp_index] = val
            feature_index += 1
        comp_index += 1

    # save image
    fig = plt.figure(frameon=False)
    fig.set_size_inches(0.06, 0.04)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(preprocessed_image, aspect='auto', cmap='Greys')
    image_path = 'D:\\saved-images\\digital_image.png'
    plt.savefig(image_path)
    plt.close(fig)

    digital_image = plt.imread(image_path)[:,:,:3]
    digital_image = np.array([digital_image])
    y_label = sample[0]['y_test']

    model_dataset_id = plpy.execute(f'select dataset_id from models_table where model_name = \'{model_name}\'')[0]['dataset_id']

    class_samples = []
    samples_train = plpy.execute(
        f'select x_train from train_table '
        f'where dataset_id = {model_dataset_id} and y_train = \'{y_label}\''
    )
    class_samples += samples_train

    samples_test = plpy.execute(
        f'select x_test from test_table '
        f'where dataset_id = {model_dataset_id} and y_test = \'{y_label}\''
    )
    class_samples += samples_test

    samples_val = plpy.execute(
        f'select x_val from val_table '
        f'where dataset_id = {model_dataset_id} and y_val = \'{y_label}\''
    )
    class_samples += samples_val

    total_class_img = [[0., 0., 0., 0., 0., 0.] for _ in range(4)]
    for sample in class_samples:
        bytes_img = None
        table_name = ''
        if sample.get('x_train') is not None:
            bytes_img = sample.get('x_train')
        elif sample.get('x_test') is not None:
            bytes_img = sample.get('x_test')
        elif sample.get('x_val') is not None:
            bytes_img = sample.get('x_val')
        array_img = pickle.loads(bytes_img)

        # get average digital image
        for i in range(len(array_img)):
            for j in range(len(array_img[0])):
                total_class_img[i][j] += array_img[i][j]

    total_class_img += digital_image
    average_img = np.true_divide(total_class_img, 151)

    predict_value = new_model.predict(average_img)
    count = 1
    for shell in predict_value:
        for value in shell:
            plpy.notice(f'{count} - {100 * value.astype(float):.5f}%')
            count += 1
    predict_result = np.argmax(predict_value)
    plpy.notice(f"predict = {predict_result + 1}")

    return f"sample_id = {test_sample_id}, predict = {predict_result + 1}"
$BODY$;

SELECT * FROM test_table;
SELECT * FROM models_table;
SELECT test_original_image('conv2d-12', 1);
