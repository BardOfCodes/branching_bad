import os
import sys
import torch as th
from wacky import load_config_file
from procXD import SketchBuilder
from branching_bad.utils.arg_parser import arg_parser
from branching_bad.utils.notification_utils import SlackNotifier
import branching_bad.meta_proc as meta_factory


def main():

    # th.autograd.set_detect_anomaly(True)
    th.backends.cudnn.benchmark = True
    try:
        th.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    args, reminder_args = arg_parser.parse_known_args()
    config = load_config_file(args.config_file, reminder_args)
    G = config.to_graph()
    save_file = os.path.join(config.LOGGER.LOG_DIR, "config.excalidraw")
    sys.path.insert(0, config.SIRI_PATH)
    sketch_builder = SketchBuilder(save_path=save_file)
    sketch_builder.render_stack_sketch(G, stacking="vertical")
    sketch_builder.export_to_file()
    del sketch_builder

    experiment_proc = getattr(meta_factory, config.EXPERIMENT_MODE)
    experiment = experiment_proc(config)
    notif = SlackNotifier(config.NAME, config.NOTIFICATION)
    try:
        notif.start_exp()
        experiment.start_experiment()
    except Exception as ex:
        notif.exp_failed(ex)
        raise ex


if __name__ == '__main__':
    main()
