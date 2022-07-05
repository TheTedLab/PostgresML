create extension plpython3u;

create table if not exists train_table
(
    id serial primary key,
    x_train numeric[28][28] not null,
    y_train integer not null
);

create table if not exists test_table
(
    id serial primary key,
    x_test numeric[28][28] not null,
    y_train integer not null
);

create or replace function tf_version()
returns text
as $$
    import tensorflow as tf
    return tf.__version__
$$
language 'plpython3u';

select tf_version();

create or replace function load_mnist()
    returns void
    language 'plpython3u'
as $BODY$
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

begin;
select load_mnist();
select * from train_table order by id;
select * from test_table order by id;
rollback;

truncate table train_table;
truncate table test_table;

create or replace function show_sample(
    sample_table text,
    sample_id integer)
    returns text
    language 'plpython3u'
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

begin;
select show_sample('train', 0);
select show_sample('test', 0);
select show_sample('train_data', 0);
select show_sample('test', 10000);
rollback;
