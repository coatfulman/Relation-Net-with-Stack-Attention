import os

import tensorflow as tf
import subprocess, tempfile

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
tf.flags.DEFINE_string("output_file", "../coco-caption/results/captions_val2014_baseline_results.json",
                       "result file.")

tf.flags.DEFINE_integer('threads', 4, "Number of threads")

tf.flags.DEFINE_string("model", "rn",
                        "Model type (rn, inception).")

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

  for _ in range(FLAGS.threads):
    if _ == FLAGS.threads-1:
      f = filenames[ n // FLAGS.threads * _ : ]
    else:
      f = filenames[ n // FLAGS.threads * _ : n // FLAGS.threads * (_+1) ]
    f = ','.join(f)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir='.')
    # tmp_file.write(FLAGS.checkpoint_path.encode("utf-8") + b',')
    # tmp_file.write(tmp_file.name.encode("utf-8") + b',')
    # tmp_file.write(FLAGS.model.encode("utf-8") + b',')
    tmp_file.write(f.encode("utf-8"))
    tmp_file.close()
    tmp_files.append(tmp_file.name)

    # cmd = ['python3', 'im2txt/coco_inference.py', '--checkpoint_path', FLAGS.checkpoint_path,
    #   '--tmpfile', tmp_file.name, '--model', FLAGS.model]
    # cmd = ['python3', 'im2txt/coco_inference.py', '--tmpfile', tmp_file.name]
    cmd = 'python3 im2txt/coco_inference.py --tmpfile=' + tmp_file.name + \
          ' --checkpoint_path=' + FLAGS.checkpoint_path + ' --model=' + FLAGS.model + \
          ' --vocab_file=' + FLAGS.vocab_file

    p = subprocess.Popen(cmd, shell=True)
    procs.append(p)

  for p in procs: p.wait()

  fo = tf.gfile.GFile(FLAGS.output_file, "w")
  fo.write('[')
  for _ in range(FLAGS.threads):
    f = open(tmp_files[_], 'r').read()
    if _ == FLAGS.threads-1: f = f[:-2]
    fo.write( f )

  fo.write(']')
  fo.close()

if __name__ == "__main__":
  tf.app.run()
