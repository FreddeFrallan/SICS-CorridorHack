# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

from Demo.Img2Txt.im2txt.im2txt import configuration, inference_wrapper
from Demo.Img2Txt.im2txt.im2txt.inference_utils import vocabulary, caption_generator
from Demo.Img2Txt import Constants

IMAGE_FILE = "C:\\Users\\Fredde\\Pictures\\fanny.jpg"

def main():
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   Constants.CHECKPOINT_PATH)
    g.finalize()


    vocab = vocabulary.Vocabulary(Constants.VOCABULARY_PATH)
    filenames = []

    for file_pattern in IMAGE_FILE.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))

    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), IMAGE_FILE)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)


        for filename in filenames:
            f = open(filename, 'rb')
            image = f.read()

            captions = generator.beam_search(sess, image)
            print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))