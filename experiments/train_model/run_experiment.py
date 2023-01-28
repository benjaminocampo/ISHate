import argparse
import os
import itertools as it
from inspect import cleandoc

printf = lambda s: print(s, flush=True)

cmd_error_count = 0


def cmd(c):
    global cmd_error_count
    red_color = "\033[91m"
    end_color = "\033[0m"
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    if os.system(c):  # Command returned != 0
        printf(
            f"{red_color}>>> [ERROR] there was an error in command:{end_color}"
        )
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        cmd_error_count += 1
        exit()


submission_text = lambda run_name: cleandoc(
    f"""
    #!/bin/bash
    #SBATCH --job-name={run_name}
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH -o ./slurmout/{run_name}.out
    #SBATCH -e ./slurmerr/{run_name}.err
    srun 
    """)

oar_submission_text = lambda run_cmd: cleandoc(
    f"""
    oarsub -p "mem > 100000" -l /nodes=1,walltime=24 --stdout=run.out --stderr=run.err -q default 'conda activate sald; {run_cmd}'
    """
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cluster-batch",
        type=str, choices=["slurm", "oar"], default=None
    )
    args = parser.parse_args()
    use_cluster_batch = args.use_cluster_batch
    tasks = ["implicit_task", "subtle_task"]
    aug_methods = [
        None, "rsa", "aav", "rne", "ri", "ra", "eda", "all", "bt", "gm",
        "gm_revised"
    ]
    models = ["bert", "deberta", "hatebert", "usesvm"]

    runs = []
    for task, model, aug_method in it.product(tasks, models, aug_methods):
        run_name = f"task-{task}__model-{model}__aug_method-{aug_method}"

        cmd_str = (submission_text(run_name) if use_cluster_batch == "slurm" else
               "") + ("python train_model.py"
                      f" train/model={model}"
                      f" input.train_file=../../data/{task}/train.parquet.gzip"
                      f" input.dev_file=../../data/{task}/dev.parquet.gzip"
                      f" input.test_file=../../data/{task}/test.parquet.gzip"
                      f" input.run_name={run_name}"
                       " input.train_size=0.05") + (
                          f" ++input.augmentation_file=../../data/{task}/aug_data/{aug_method}.parquet.gzip"
                          if aug_method is not None else "")
        runs.append((run_name, cmd_str))

    if use_cluster_batch == "slurm":
        os.makedirs("./slurmerr", exist_ok=True)
        os.makedirs("./slurmout", exist_ok=True)
        os.makedirs("./bash_cmds", exist_ok=True)
        for run_name, cmd_str in runs:
            with open(f"./bash_cmds/{run_name}.sh", "w") as submission:
                submission.write(cmd_str)
            cmd(f"sbatch ./bash_cmds/{run_name}.sh")
    elif use_cluster_batch == "oar":
        for run_name, cmd_str in runs:
            cmd(oar_submission_text(cmd_str))
    else:
        for run_name, cmd_str in runs:
            cmd(cmd_str)


if __name__ == "__main__":
    main()
