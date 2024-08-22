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
        "--npf_folder",
        "-npf_f",
        type=str,
        default="/vol/bitbucket/ad6013/Research/Neural-Process-Family/",
        help="Folder where the Neural Process Family is stored.",
    )
    parser.add_argument(
        "--work_dir",
        "-wd",
        type=str,
        default="/vol/bitbucket/ad6013/Research/ml2_meta_causal_discovery/ml2_meta_causal_discovery/",
        help="Folder where the Neural Process Family is stored.",
    )
    parser.add_argument(
        "--learning_rate_min",
        "-lr_min",
        type=float,
        default=5e-4,
        help="Learning rate minimum for optimizer.",
    )
    parser.add_argument(
        "--learning_rate_max",
        "-lr_max",
        type=float,
        default=0.001,
        help="Learning rate maximum for optimizer.",
    )
    parser.add_argument(
        "--optimizer",
        "-opt",
        type=str,
        default="Adam",
        help="Optimizer from torch.optim.",
    )
    parser.add_argument(
        "--scheduler",
        "-sch",
        type=str,
        default="None",
        help="Scheduler from torch.optim.lr_scheduler.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=32,
        help="Batch size for generator.",
    )
    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=1000,
        help="Number of samples per function.",
    )
    parser.add_argument(
        "--max_context_size",
        "-maxcs",
        type=int,
        default=900,
        help="Maximum number of context points.",
    )
    parser.add_argument(
        "--min_context_size",
        "-mincs",
        type=int,
        default=100,
        help="Minimum nuber of context points.",
    )
    parser.add_argument(
        "--max_epochs",
        "-me",
        type=int,
        default=400,
        help="Max epochs to run for.",
    )
    parser.add_argument(
        "--epoch_steps_train",
        "-est",
        type=int,
        default=1000,
        help="Number of data iterations that defines an epoch.",
    )
    parser.add_argument(
        "--epoch_steps_validation",
        "-esv",
        type=int,
        default=5,
        help="Number of data iterations that defines an epoch.",
    )
    parser.add_argument(
        "--only_xcause_y",
        action="store_true",
        default=False,
        help="Number of data iterations that defines an epoch.",
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
        "--vi_train",
        "-vi",
        action="store_true",
        default=False,
        help="Whether to train LNP models using VI else will use ML.",
    )
    parser.add_argument(
        "--nz_samples_train",
        "-nz_train",
        type=int,
        default=16,
        help="Number of samples for training LNP models.",
    )
    parser.add_argument(
        "--num_encoder_hidden_layers",
        "-nehl",
        type=int,
        default=5,
        help="Number of hidden layers in the encoder.",
    )
    parser.add_argument(
        "--mlflow",
        "-ml",
        action="store_true",
        default=False,
        help="Whether to use MLFLOW.",
    )
    parser.add_argument(
        "--mlp_encoder",
        "-mlpenc",
        action="store_true",
        default=False,
        help="Whether to use MLP encoder.",
    )
    parser.add_argument(
        "--attention_encoder",
        "-attenc",
        action="store_true",
        default=False,
        help="Whether to use Attention Encoder.",
    )
    parser.add_argument(
        "--mlp_decoder",
        "-mlpdec",
        action="store_true",
        default=False,
        help="Whether to use MLP decoder.",
    )
    parser.add_argument(
        "--attention_decoder",
        "-attdec",
        action="store_true",
        default=False,
        help="Whether to use Attention Decoder.",
    )
    parser.add_argument(
        "--gamma1",
        "-g1",
        default=1.0,
        type=float,
        help="Gamma 1 for the data generation.",
    )
    parser.add_argument(
        "--gamma2",
        "-g2",
        default=2.0,
        type=float,
        help="Gamma 2 for the data generation.",
    )
    parser.add_argument(
        "--kernel_sum",
        "-ks",
        action="store_true",
        default=False,
        help="Whether to sum all kernels in data generation.",
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
        "--normal_prior",
        "-np",
        default=False,
        action="store_true",
        help="Whether to use P(z) as a normal distribution without any encoder",
    )
    parser.add_argument(
        "--bimodal_prior",
        "-bimp",
        default=False,
        action="store_true",
        help="Whether to use a bimodal prior without any encoder.",
    )
    parser.add_argument(
        "--classify_run_name",
        "-crn",
        type=str,
        default="test",
        help="Run name for saving and MLFlow classifier experiment.",
    )
    parser.add_argument(
        "--classify_experiment_name",
        "-cen",
        type=str,
        default="test",
        help="Experiment name for saving and MLFlow classifier experiment.",
    )
    parser.add_argument(
        "--classify_np_end_to_end",
        "-clnpetoe",
        default=False,
        action="store_true",
        help="Whether to train the classifier end to end with the NP.",
    )

    parser.add_argument(
        "--only_observational",
        "-only_obs",
        default=False,
        action="store_true",
        help="Only use observational data.",
    )
    parser.add_argument(
        "--only_interventional",
        "-only_int",
        default=False,
        action="store_true",
        help="Only use interventional data.",
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
        "--num_datasets",
        "-nd",
        default=1000000,
        type=int,
        help="Number of datasets to use for training.",
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
