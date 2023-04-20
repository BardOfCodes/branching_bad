import argparse

arg_parser = argparse.ArgumentParser(description="singular parser")
arg_parser.add_argument('--config-file', type=str,
                        default="configs/config.yml")
arg_parser.add_argument('--debug', type=bool, default=False)
