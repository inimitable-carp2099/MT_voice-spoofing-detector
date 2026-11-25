# Runs files 01-08 in one namespace and calls the requested step.
#   python run.py train                     train, save best and final checkpoints
#   python run.py evaluate                  select checkpoint, EER on four test sets
#   python run.py decide path/to/file.flac  decision for a single recording
# Set dataset and model paths in 01_config.py first.
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob.glob(os.path.join(HERE, "0[1-8]_*.py")))


def load():
    ns = {"__name__": "detector"}
    for path in FILES:
        with open(path, encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), ns)
    return ns


def main(argv):
    if len(argv) < 2 or argv[1] not in ("train", "evaluate", "decide"):
        print("usage: python run.py train | evaluate | decide <audio>")
        return
    d = load()
    if argv[1] == "train":
        d["train"]()
    elif argv[1] == "evaluate":
        best_ckpt, val_eers = d["select_checkpoint"]([d["BEST_CKPT_PATH"], d["FINAL_CKPT_PATH"]])
        print("checkpoint:", best_ckpt, val_eers)
        model = d["load_model"](best_ckpt)
        for name in ("2019", "2021", "ITW", "ASV5"):
            result, _ = d["evaluate_test_set"](model, name)
            print(name, result["EER"])
    else:
        model = d["load_model"](d["FINAL_CKPT_PATH"])
        theta = d["dev_threshold"](model)
        _, itw = d["evaluate_test_set"](model, "ITW")
        val_mask, _ = d["itw_speaker_halves"](itw["group"].values)
        decider = d["SpoofDecision"].fit(theta, itw["score"].values[val_mask], itw["u"].values[val_mask],
                                         criterion="u", q=10)
        s, u = d["score_file"](model, argv[2])
        print(decider.decide([s], [u])[0])


if __name__ == "__main__":
    main(sys.argv)
