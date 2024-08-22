"""
This file will contain all the argparse arguements.
"""


def retun_default_args(parser):
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--work_dir",
        "-wd",
        type=str,
        default="/vol/bitbucket/ad6013/Research/CausalStructureNeuralProcess/ml2_meta_causal_discovery/",
        help="Folder where the Neural Process Family is stored.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2e-4,
        help="Learning rate minimum for optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        "-wdecay",
        type=float,
        default=0.00001,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--optimizer",
        "-opt",
        type=str,
        default="AdamW",
        help="Optimizer from torch.optim.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=32,
        help="Batch size for generator.",
    )
    parser.add_argument(
        "--max_epochs",
        "-me",
        type=int,
        default=2,
        help="Max epochs to run for.",
    )
    parser.add_argument(
        "--run_name",
        "-rn",
        type=str,
        default="test",
        help="Run name for saving and MLFlow.",
    )
    parser.add_argument(
        "--experiment_name",
        "-en",
        type=str,
        default="test",
        help="Experiment name for saving and MLFlow.",
    )
    parser.add_argument(
        "--data_file",
        "-df",
        type=str,
        default="gplvm_causal_graphs",
        help="File where data is stored.",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=10,
        help="Number of workers to load the data.",
    )
    parser.add_argument(
        "--num_layers_encoder",
        "-nle",
        default=5,
        type=int,
        help="Number of layers in the encoder.",
    )
    parser.add_argument(
        "--num_layers_decoder",
        "-nde",
        default=2,
        type=int,
        help="Number of layers in the decoder.",
    )
    parser.add_argument(
        "--dim_model",
        "-dm",
        default=128,
        type=int,
        help="Hidden dims.",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        "-lrw_ratio",
        default=0.1,
        type=float,
        help="Ratio of the total number of steps to use in warmup.",
    )
    parser.add_argument(
        "--decoder",
        "-dec",
        type=str,
        help="Decoder to use. [autoregressive, probabilistic, transformer]",
    )
    parser.add_argument(
        "--dim_feedforward",
        "-dim_ff",
        default=2048,
        type=int,
        help="Feedforward dimension in the transformer.",
    )
    parser.add_argument(
        "--num_nodes",
        "-nnodes",
        required=True,
        type=int,
        help="Number of nodes in the graph.",
    )
    parser.add_argument(
        "--nhead",
        "-head",
        default=8,
        type=int,
        help="Number of heads in the transformer.",
    )
    parser.add_argument(
        "--n_perm_samples",
        "-nps",
        default=25,
        type=int,
        help="Number of samples for the permutations.",
    )
    parser.add_argument(
        "--sinkhorn_iter",
        "-si",
        default=300,
        type=int,
        help="Number of sinkhorn iterations.",
    )
    parser.add_argument(
        "--use_positional_encoding",
        "-upe",
        default=False,
        action="store_true",
        help="Use positional encoding in the transformer.",
    )


    args = parser.parse_args()
    return args
