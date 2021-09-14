from transformers import BertJapaneseTokenizer, BertForSequenceClassification, PretrainedBartModel, CamembertTokenizer, RobertaForSequenceClassification
import os, sys
sys.path.append(os.pardir)
from utils.parser import parse_args

def load_model(model_name: str):
    r"""
    load model

    :param model_name: options: tohoku_bert, aozora_wiki_bert, aozora_wiki_bert_tapt
    :return: model and tokenizer
    """

    args = parse_args()

    if model_name == "HAN":
        data_dir = Path(f"{args.data_dir}/{args.model_type}")
        LOG_DIR = Path(f"{args.log_dir}/{args.model_type}")
        WEIGHT_DIR = LOG_DIR / "weights"
        MODEL_FILE = WEIGHT_DIR / args.best_model
        EMBEDDING_MATRIX = Vectors(name=data_dir / "emb/emb.vec", cache=data_dir / "cache")
        model = HierAttnNet(
            vocab_size=len(EMBEDDING_MATRIX.vectors),
            word_hidden_dim=args.word_hidden_dim,
            sent_hidden_dim=args.sent_hidden_dim,
            padding_idx=args.padding_idx,
            embed_dim=args.embed_dim,
            weight_drop=args.weight_drop,
            embed_drop=args.embed_drop,
            locked_drop=args.locked_drop,
            last_drop=args.last_drop,
            embedding_matrix=EMBEDDING_MATRIX,
            num_class=2 if args.model_type == "binary" else 7,
            device=args.gpu_num
        )
        model.load_state_dict(torch.load(MODEL_FILE), strict=False)
        #model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

        model = model.to(DEVICE)
        return model

    elif model_name == "sentence_classifier_RoBERTa":
        tokenizer_path = "/home/itsunoda/デスクトップ/vscode_workspace/PYTHON/sotuken/tokenizing/tokenizer"
        model_path = "/home/itsunoda/デスクトップ/vscode_workspace/PYTHON/sotuken/corpus/dataset_for_fine-tuning/categorized_level-0_with_others/254-model-epoch-3"
        tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path)
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
        return model, tokenizer

    else:
        raise ValueError(f"model_name: {model_name} is invalid.")
