# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "model/train/",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("tmpfile", "../coco-caption/results/tmp",
                       "tmp file for image file ids.")
# tf.flags.DEFINE_string("filepath", "../val2014/COCO_val2014_000000??????.jpg", "Text file containing the vocabulary.")
filepath = "../data/val-?????-of-00004"

tf.logging.set_verbosity(tf.logging.ERROR)

# import sys
# print (sys.argv)

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  # filenames = []
  # for file_pattern in FLAGS.input_files.split(","):
  #   filenames.extend(tf.gfile.Glob(file_pattern))
  # tf.logging.info("Running caption generation on %d files matching %s",
  #                 len(filenames), FLAGS.input_files)

  # filenames = tf.gfile.GFile(FLAGS.tmpfile, "r").read().split(',')
  # tf.logging.info("Running caption generation on %d", len(filenames))
  reader = tf.TFRecordReader()
  input_queue = input_ops.prefetch_input_data(
          reader,
          filepath,
          is_training=False,
          batch_size=1,
          values_per_shard=2300,
          input_queue_capacity_factor=2,
          num_reader_threads=1)

  fo = tf.gfile.GFile(FLAGS.tmpfile, "w")
  # fo.write('[')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=g, config=config) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    num_eval_examples = 10132
    for _ in range(num_eval_examples):
      serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
      captions = generator.beam_search(sess, image)
      # print("Captions for image %s:" % os.path.basename(filename))
      
      caption = captions[0]
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      if sentence[-1] == '.': sentence = sentence[:-1]
      sentence = ' '.join(sentence)

      # image_id = filename[-10:-4].lstrip('0')
      image_id = filename.lstrip('0')
      sentence = '{\"image_id\": %s, \"caption\": \"%s\"}, ' % ( image_id, sentence )
      fo.write(sentence)
      # for i, caption in enumerate(captions):
        # Ignore begin and end words.
        # sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        # sentence = " ".join(sentence)
        # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
      _ += 1
      if _%100 == 0: print (FLAGS.tmpfile, _)

  # fo.write(']')
  fo.close()

if __name__ == "__main__":
  tf.app.run()
