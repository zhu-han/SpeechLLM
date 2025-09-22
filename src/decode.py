#!/usr/bin/env python3

"""
Usage:
# Command for decoding LLM-based ASR models with a single speech encoders:

python3 ./src/decode.py \
  --max-duration 2000 \
  --exp-dir exp/asr_librispeech_zipformer_qwen2_1.5B \
  --epoch 10 --avg 5 \
  --manifest-dir data/fbank
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import transformers
from lhotse.cut import Cut
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr_datamodule import AsrDataModule, LibriSpeechDataset
from model import EncoderProjector, Speech_LLM_Zipformer
from train import (
    DEFAULT_SPEECH_TOKEN,
    END_SPEECH_TOKEN,
    END_TEXT_TOKEN,
    START_SPEECH_TOKEN,
    START_TEXT_TOKEN,
    add_model_arguments,
)
from utils import (
    AttributeDict,
    average_checkpoints,
    get_env_info,
    setup_logger,
    store_transcripts,
    write_error_stats,
)
from zipformer.load import ZipformerEncoderModel, get_zipformer_model


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="""It specifies the checkpoint to use for decoding, 
        e.g., epoch-10-avg-5.pt. Will ignore --epoch and --avg if it is set.""",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="beam size for beam search decoding",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/asr_librispeech_zipformer_qwen2_1.5B",
        help="The experiment dir",
    )

    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "env_info": get_env_info(),
        }
    )
    return params


def decode_one_batch(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    batch: dict,
) -> List[List[str]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: "beam-search"
        - value: A list of lists. Each sublist is a list of token IDs.
    Args:
        params:
            It is returned by :func:`get_params`.
        model:
            The neural model.
        batch:
            It is returned by :meth:`torch.utils.data.DataLoader.__iter__`.
    Returns:
        Return a list.
    """

    def preprocess(
        messages,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocesses the data for supervised fine-tuning."""
        texts = []
        TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        for i, msg in enumerate(messages):
            texts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=True,
                    add_generation_prompt=False,
                    chat_template=TEMPLATE,
                    padding=False,
                    truncation=True,
                )
            )
        max_len_texts = max([len(text) for text in texts])
        # left padding texts to the same length, texts is a list of list,
        # padding with tokenzier.pad_token_id
        texts = [
            [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
            for text in texts
        ]

        input_ids = torch.tensor(texts, dtype=torch.int)

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        return input_ids, attention_mask

    device = model.llm.device

    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device, dtype=torch.float32)

    supervisions = batch["supervisions"]
    feature_len = supervisions["num_frames"]
    feature_len = feature_len.to(device)

    speech_part = f"{START_SPEECH_TOKEN}{DEFAULT_SPEECH_TOKEN}{END_SPEECH_TOKEN}"
    messages = [
        [
            {
                "role": "user",
                "content": speech_part,
            },
            {"role": "assistant", "content": ""},
        ]
    ] * len(feature)

    input_ids, attention_mask = preprocess(messages, tokenizer)

    generated_ids = model.decode(
        feature,
        feature_len,
        input_ids.to(device, dtype=torch.long),
        attention_mask.to(device),
    )
    hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return hyps


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    tokenizer: AutoTokenizer,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
        dl:
            The dataloader.
        params:
            It is returned by :func:`get_params`.
        model:
            The neural model.
    Returns:
        Return a list.
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = list()
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps = decode_one_batch(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
        )

        assert len(hyps) == len(texts)
        for cut_id, hyp_units, ref_text in zip(cut_ids, hyps, texts):
            hyp_text = "".join(hyp_units)
            logging.debug(f"ref: {ref_text}")
            logging.debug(f"hyp: {hyp_text}")
            results.append((cut_id, ref_text.split(), hyp_text.split()))

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
):

    recog_path = params.decode_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    results = sorted(results)
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = params.decode_dir / f"errs-{test_set_name}-{params.suffix}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    errs_info = params.decode_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    s = f"\nFor {test_set_name} WER: {wer}"
    with open(errs_info, "w") as f:
        print(s, file=f)

    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.decode_dir = Path(f"{params.exp_dir}/decode")
    params.suffix = (
        f"decode-beam{params.beam_size}-epoch-{params.epoch}-avg-{params.avg}"
    )

    setup_logger(f"{params.decode_dir}/log-decode-{params.suffix}")

    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"device: {device}")

    zipformer_model = get_zipformer_model(
        params.speech_encoder_type,
        params.speech_encoder_path,
        params.speech_encoder_bpe_path,
        "cpu",
    )
    speech_encoder = ZipformerEncoderModel(
        zipformer_model.encoder_embed, zipformer_model.encoder
    )
    speech_encoder_dim = max(speech_encoder.encoder.encoder_dim)
    tokenizer = AutoTokenizer.from_pretrained(params.llm_path)

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    if params.use_lora:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()

    special_tokens_dict = {
        "additional_special_tokens": [
            DEFAULT_SPEECH_TOKEN,
            START_TEXT_TOKEN,
            END_TEXT_TOKEN,
            START_SPEECH_TOKEN,
            END_SPEECH_TOKEN,
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    llm.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert tokenizer.pad_token_id == llm.config.pad_token_id
    llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )
    llm.config.start_text_token_id = tokenizer.convert_tokens_to_ids(START_TEXT_TOKEN)
    llm.config.end_text_token_id = tokenizer.convert_tokens_to_ids(END_TEXT_TOKEN)
    llm.config.start_speech_token_id = tokenizer.convert_tokens_to_ids(
        START_SPEECH_TOKEN
    )
    llm.config.end_speech_token_id = tokenizer.convert_tokens_to_ids(END_SPEECH_TOKEN)

    encoder_projector = EncoderProjector(
        speech_encoder_dim, llm.config.hidden_size, params.encoder_projector_ds_rate
    )

    model = Speech_LLM_Zipformer(
        speech_encoder,
        llm,
        encoder_projector,
    )

    if params.checkpoint is not None:
        logging.info(f"Decoding using checkpoint {params.exp_dir}/{params.checkpoint}")
        checkpoint = torch.load(
            f"{params.exp_dir}/{params.checkpoint}",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        if params.avg > 1:
            start = params.epoch - params.avg + 1
            assert start >= 1, start
            checkpoint = torch.load(
                f"{params.exp_dir}/epoch-{params.epoch}.pt",
                map_location="cpu",
                weights_only=False,
            )
            filenames = [
                f"{params.exp_dir}/epoch-{epoch}.pt"
                for epoch in range(start, params.epoch + 1)
            ]
            avg_checkpoint = average_checkpoints(filenames)
            model.load_state_dict(avg_checkpoint, strict=False)

            filename = f"{params.exp_dir}/epoch-{params.epoch}-avg-{params.avg}.pt"
            torch.save({"model": avg_checkpoint}, filename)
            logging.info(f"Decoding using checkpoint {filename}")
        else:
            logging.info(
                f"Decoding using checkpoint {params.exp_dir}/epoch-{params.epoch}.pt"
            )
            checkpoint = torch.load(
                f"{params.exp_dir}/epoch-{params.epoch}.pt",
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True

    data_module = AsrDataModule(args)
    multi_dataset = LibriSpeechDataset(args.manifest_dir)

    test_sets_cuts = multi_dataset.librispeech_test_cuts()

    test_sets = test_sets_cuts.keys()
    test_dls = [
        data_module.test_dataloaders(test_sets_cuts[cuts_name])
        for cuts_name in test_sets
    ]

    for test_set, test_dl in zip(test_sets, test_dls):
        results = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            tokenizer=tokenizer,
        )

        save_results(params=params, test_set_name=test_set, results=results)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
