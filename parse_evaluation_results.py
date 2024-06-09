import pandas as pd
from absl import flags, app

import util

FLAGS = flags.FLAGS

flags.DEFINE_string("results_file", None, "Path to the results file that should be parsed.")


def main(_):
    df, best_model_name, rmse = util.parse_evaluation_results(FLAGS.results_file)
    print(f'Models evaluated: {", ".join(df.columns)}')
    print(f'Best model at p95: {best_model_name}, p95 RMSE: {rmse}')


if __name__ == "__main__":
    app.run(main)
