import os
import tempfile

from IPython import embed


sbatch_template = """
#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__save_dir__/__job_name__.out
#SBATCH -e __save_dir__/__job_name__.err
#SBATCH --partition=superpod-a100    # gpu
#
#SBATCH -n 8
#SBATCH -G 1     # --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

export PYTHONPATH="${PYTHONPATH}:/work/pi_mccallum_umass_edu/rangell_umass_edu/chess-gpt-circuits/"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate sae

python circuits/train_saes_parallel.py --game __game__ --layer __layer__ --trainer_type __trainer_type__ --save_dir __save_dir__ __extra_args__
"""


if __name__ == "__main__":

    # set the args
    dry_run = False
    trained_model = False
    game = "othello"
    layer = 5
    trainer_type = "standard"
    save_dir = f"circuits/dictionary_learning/dictionaries/{game}-"\
               f"{'trained_model' if trained_model else 'random_model'}-"\
               f"layer_{layer}-{trainer_type}"

    extra_args = []
    if dry_run:
        print("WARNING: dry run is set to true -- not training!!!")
        extra_args.append("--dry_run")
    if not trained_model:
        extra_args.append("--random_model")
    extra_args = " ".join(extra_args)

    # fill out the job template
    sbatch_str = sbatch_template
    sbatch_str = sbatch_str.replace("__job_name__", trainer_type)
    sbatch_str = sbatch_str.replace("__save_dir__", save_dir)
    sbatch_str = sbatch_str.replace("__game__", game)
    sbatch_str = sbatch_str.replace("__layer__", str(layer))
    sbatch_str = sbatch_str.replace("__trainer_type__", trainer_type)
    sbatch_str = sbatch_str.replace("__extra_args__", extra_args)
    sbatch_str = sbatch_str.strip()

    submit = True
    # submit the job
    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(sbatch_str.strip(), "utf-8"))
        f.seek(0)
        if submit:
            os.system(f"sbatch {f.name}")
            

