from dataclasses import dataclass, field


@dataclass
class BaseArguments:
    wandb_project_name: str = field(
        default='graph_lora',
        metadata={'help': "The name of the project for wandb logging."}
    )

    wandb_run_name: str = field(
        default='test',
        metadata={'help': "The name of the run for wandb logging."}
    )

    input_file: str = field(
        default="outputs/data/scene_graph/processed_test.json",
        metadata={'help': "The input file for experiment."}
    )

    ref_file: str = field(
        default="train.json",
        metadata={"help": "the file path of reference data to be used for sampling example questions."}
    )

    output_file: str = field(
        default="outputs/lora_inf/scene_graph/test_mp.json",
        metadata={'help': "The output file for writing results"}
    )

    overwrite: bool = field(
        default=True,
        metadata={'help': "Whether to overwrite the output file"}
    )

    rank: int = field(
        default=0,
        metadata={'help': "The rank of the process"}
    )
