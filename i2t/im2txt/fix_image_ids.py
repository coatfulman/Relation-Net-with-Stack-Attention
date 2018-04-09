import os

import tensorflow as tf
import subprocess, tempfile
import json

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "model/train/",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "../val2014/COCO_val2014_000000??????.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
# tf.flags.DEFINE_string("input_folder", "../val2014/COCO_val2014_000000??????.jpg",
#                        "File pattern or comma-separated list of file patterns "
#                        "of image files.")
tf.flags.DEFINE_string("output_file", "../coco-caption/results/captions_val2014_baseline_results.json.bad",
                       "result file.")

tf.flags.DEFINE_integer('threads', 4, "Number of threads")

tf.logging.set_verbosity(tf.logging.ERROR)


def main(_):
  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  n = len(filenames)
  tf.logging.info("Running caption generation on %d files matching %s with %d threads\n",
                  n, FLAGS.input_files, FLAGS.threads)

  input_pattern = FLAGS.input_files.replace('??????', '%s')
  filenames = [f[-10:-4] for f in filenames]

  tmp_files = []
  procs = []

  dataset = open(FLAGS.output_file, 'r').read()[1:-1]
  dataset = dataset.split('},')

  print (dataset[12])

  dataset = [ '{\"image_id\": %s, \"%s},' % ( image_id.lstrip('0'), sentence.split(', \"')[1] )
              for image_id, sentence in zip(filenames, dataset)]
  print (dataset[12])

  open(FLAGS.output_file[:-4], 'w').write('[' + '\n'.join(dataset)[:-2] + ']')
  # json.dump(dataset, open(FLAGS.output_file + 'fixed', 'w'))

if __name__ == "__main__":
  tf.app.run()
