import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run role estimation of wereWolf")

    parser.add_argument("--model_type", type=str, default="multi", help="Select binary, multi or pure.")
    parser.add_argument("--b_role_dict", type=dict, default={0: "市民陣営", 1: "人狼陣営"})
    parser.add_argument("--m_role_dict", type=dict, default={0: "人狼", 1: "狂人", 2: "村人", 3: "占い師", 4: "霊能者", 5: "狩人"})
    parser.add_argument("--English_m_role_dict", type=dict, default={0: "Werewolf", 1: "Lunatic", 2: "Villager", 3: "Seer", 4: "Medium", 5: "Hunter"})

    parser.add_argument("--gpu_num", type=int, default=0, help="The number of GPU")
    #Configuration of Distributed Data Palallel.
    #parser.add_argument("--local_rank", type=int, default=0)
    #parser.add_argument("--local_world_size", type=int, default=1)

    parser.add_argument("--base_dir", type=str, default="../")
    parser.add_argument("--han_dir", type=str, default="../HAN")
    parser.add_argument("--data_dir", type=str, default="../HAN/data", help="Input data path.")
    parser.add_argument("--log_dir", type=str, default="../HAN/results", help="Store model path.")

    # mkdata
    parser.add_argument("--exclude", action="store_true", help="Whether exclude sentences which have peculiar wereWolf statements or not.")
    parser.add_argument("--auto", action="store_true", help="Whether exclude the sentences automatically or manually.")
    parser.add_argument("--sample", action="store_true")


    # Set model
    parser.add_argument("--model", type=str, default="han", help="Specify the model han or ...")
    parser.add_argument("--best_model", type=str, default="han lr:0.001, wdc:0.01, bsz:64, whd:64, shd:64, emb:300, ldrp:0.2, sch:no.pt")
    #  Parameters that are common to HierAttnNet
    parser.add_argument(
        "--padding_idx",
        type=int,
        default=1,
        help="padding index for the numericalised token sequences.",
    )
    parser.add_argument(
        "--zero_padding", action="store_true", help="Manually zero padding idx when using mxnet.",
    )
    parser.add_argument("--embed_dim", type=int, default=300, help="input embeddings dimension.")
    parser.add_argument(
        "--embed_drop",
        type=float,
        default=0.2,
        help="embeddings dropout. Taken from the awd-lstm lm from Salesforce: https://github.com/salesforce/awd-lstm-lm",
    )
    parser.add_argument(
        "--locked_drop",
        type=float,
        default=0.2,
        help="embeddings dropout. Taken from the awd-lstm lm from Salesforce: https://github.com/salesforce/awd-lstm-lm",
    )
    parser.add_argument(
        "--last_drop",
        type=float,
        default=0.2,
        help="dropout before the last fully connected layer (i.e. the prediction layer)",
    )
    parser.add_argument(
        "--embedding_matrix", type=str, default=None, help="path to the pretrained word vectors.",
    )

    # HAN parameters
    parser.add_argument(
        "--word_hidden_dim",
        type=int,
        default=64,
        help="hidden dimension for the GRU processing words.",
    )
    parser.add_argument(
        "--sent_hidden_dim",
        type=int,
        default=64,
        help="hidden dimension for the GRU processing senteces.",
    )
    parser.add_argument(
        "--weight_drop",
        type=float,
        default=0.2,
        help="weight dropout. Taken from the awd-lstm lm from Salesforce: https://github.com/salesforce/awd-lstm-lm",
    )

    # Train/Test parameters
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epoch.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="l2 reg.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="no",

        help="Specify the lr_scheduler {multifactorscheduler, reducelronplateau, cycliclr, no (nothing)}",
    )
    parser.add_argument(
        "--n_cycles", type=int, default=1, help="number of cycles when using cycliclr"
    )
    parser.add_argument("--save_results", default=True, action="store_true", help="Save model and results")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--patience", type=int, default=2, help="Patience for early stopping")
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr",
    )
    parser.add_argument(
        "--steps_epochs",
        type=str,
        default="[2,4,6]",
        help="list of steps to schedule a change for the multifactorscheduler scheduler",
    )

    return parser.parse_args()
