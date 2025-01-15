import argparse
from pathlib import Path
import shutil

import yaml

from evaluator import MNISTEvaluator
from prescriptor import DeepNNPrescriptorFactory
from presp.evolution import Evolution


def main():
    """
    Main logic for running neuroevolution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=True)
    args = parser.parse_args()
    for config_path in args.config:
        print(f"Running evolution with config: {config_path}")
        with open(Path(config_path), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            save_path = Path(config["evolution_params"]["save_path"])
            if save_path.exists():
                inp = input(f"Save path {save_path} already exists. Do you want to overwrite? [Y|n].")
                if inp.lower() != "y":
                    print("Exiting.")
                    break
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True, exist_ok=False)
            with open(save_path / "config.yml", "w", encoding="utf-8") as w:
                yaml.dump(config, w)

            prescriptor_factory = DeepNNPrescriptorFactory(**config["prescriptor_params"])
            evaluator = MNISTEvaluator(**config["eval_params"])
            evolution = Evolution(prescriptor_factory=prescriptor_factory,
                                  evaluator=evaluator,
                                  **config["evolution_params"])
            evolution.run_evolution()


if __name__ == "__main__":
    main()
