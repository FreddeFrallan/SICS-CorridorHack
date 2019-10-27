'''
As expected tensorflow does a bad job version managing. So there might be a need for updating old checkpoints files.
Especially if the checkpoint is some pre-trained checkpoint found on the internet.

This simple function should solve this...
'''


def updateCheckpointFile(pathToOldCheckpoint, pathToUpdatedCheckpoint):
    import tensorflow as tf
    vars_to_rename = {
        "lstm/BasicLSTMCell/Linear/Matrix": "lstm/basic_lstm_cell/kernel",
        "lstm/BasicLSTMCell/Linear/Bias": "lstm/basic_lstm_cell/bias",
    }

    new_checkpoint_vars = {}
    reader = tf.train.NewCheckpointReader(pathToOldCheckpoint)

    for old_name in reader.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, pathToUpdatedCheckpoint)
