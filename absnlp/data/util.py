import os
import torch
import logging
from torch.utils.data import TensorDataset

from absnlp.util.ner import read_examples_from_file, convert_examples_to_features, to_features_with_vocab

logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):

    # Load data features from cache or dataset file
    dataset_path = os.path.join(args.cache_dir, args.dataset)
    cached_features_file = os.path.join(
        dataset_path,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", dataset_path)
        examples = read_examples_from_file(dataset_path, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)
    return dataset

def load_dataset_with_vocab(args, vocab, labels, pad_token_label_id, mode):
    dataset_path = os.path.join(args.cache_dir, args.dataset)
    cached_features_file = os.path.join(
        dataset_path,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    # if os.path.exists(cached_features_file) and not args.overwrite_cache:
    #     logger.info("Loading features from cached file %s", cached_features_file)
    #     features = torch.load(cached_features_file)
    # else:
    logger.info("Creating features from dataset file at %s", dataset_path)
    examples = read_examples_from_file(dataset_path, mode)
    features = to_features_with_vocab(
        examples,
        labels,
        args.max_seq_length,
        vocab,
        pad_token_label_id
    )
    
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_label_ids)
    return dataset