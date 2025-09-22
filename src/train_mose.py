#!/usr/bin/env python3
"""
Train LLM-based ASR model that combines multiple Zipformer speech encoder and Qwen2 LLM.

Usage:
# Command for downloading pre-trained Zipformer and Qwen2 models:

mkdir -p models/qwen2-1.5b-instruct
huggingface-cli download  --local-dir models/qwen2-1.5b-instruct     Qwen/Qwen2-1.5B-Instruct

mkdir -p models/zipformer/librispeech/transducer_crctc_large \
        models/zipformer/librispeech/aed_crctc_large \
        models/zipformer/librispeech/crctc_large \
        models/zipformer/librispeech/transducer_large

wget -P models/zipformer/librispeech/transducer_crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech/aed_crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-aed-20241020/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech/crctc_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-20241018/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech/transducer_large \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16/resolve/main/exp/pretrained.pt

wget -P models/zipformer/librispeech \
    https://hf-mirror.com/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/resolve/main/data/lang_bpe_500/bpe.model

# Command for training:

python3 ./zipformer_llm_en/train_mose.py \
  --world-size 4 \
  --max-duration 200 \
  --num-epochs 10 \
  --exp-dir ./zipformer_llm_en/exp_mose \
  --manifest-dir data/fbank
"""

import argparse
import logging
import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr_datamodule import AsrDataModule, LibriSpeechDataset
from model import IGNORE_TOKEN_ID, EncoderProjector, Speech_LLM_Zipformer_MoSE
from utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    get_env_info,
    get_parameter_groups_with_lrs,
    load_checkpoint,
    raise_grad_scale_is_too_small_error,
    register_inf_check_hooks,
)
from utils import save_checkpoint as save_checkpoint_impl
from utils import setup_dist, setup_logger, str2bool, torch_autocast
from zipformer.load import ZipformerEncoderModel, get_zipformer_model
from zipformer.optim import FixedLRScheduler, LRScheduler, ScaledAdam

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]

DEFAULT_SPEECH_TOKEN = "<speech>"
START_TEXT_TOKEN = "<start_text>"
END_TEXT_TOKEN = "<end_text>"
START_SPEECH_TOKEN = "<start_speech>"
END_SPEECH_TOKEN = "<end_speech>"


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--llm-path",
        type=str,
        default="models/qwen2-1.5b-instruct",
        help="Path or name of the large language model.",
    )

    parser.add_argument(
        "--speech-encoder-type",
        type=str,
        default="large,large,large,large",
        help="Types of the speech encoders, seperated with comma.",
    )

    parser.add_argument(
        "--speech-encoder-path",
        type=str,
        default="models/zipformer/librispeech/transducer_large/pretrained.pt,"
        + "models/zipformer/librispeech/aed_crctc_large/pretrained.pt,"
        + "models/zipformer/librispeech/crctc_large/pretrained.pt,"
        + "models/zipformer/librispeech/transducer_crctc_large/pretrained.pt",
        help="Path of the speech encoders, seperated with comma.",
    )

    parser.add_argument(
        "--speech-encoder-bpe-path",
        type=str,
        default="models/zipformer/librispeech/bpe.model,"
        + "models/zipformer/librispeech/bpe.model,"
        + "models/zipformer/librispeech/bpe.model,"
        + "models/zipformer/librispeech/bpe.model",
        help="Path of the bpe models of speech encoders, seperated with comma.",
    )

    parser.add_argument(
        "--encoder-projector-ds-rate",
        type=int,
        default=4,
        help="Downsample rate for the encoder projector.",
    )

    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=True,
        help="Whether to use lora to fine-tune llm.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.01, help="The base learning rate."
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer_qwen_en/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="""The path to the pretrained model if it is not None. Training will
        start from this model. e.g. ./zipformer_llm_en/exp/epoch-4-avg-3.pt
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--unfreeze-llm",
        type=str2bool,
        default=True,
        help="Whether to unfreeze llm during training.",
    )

    parser.add_argument(
        "--unfreeze-encoder",
        type=str2bool,
        default=False,
        help="Whether to unfreeze encoder during training.",
    )
    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - frame_shift_ms: The frame shift in milliseconds.
        - allowed_excess_duration_ratio: The allowed excess duration ratio.
        - best_train_loss: The best training loss so far.
        - best_valid_loss: The best validation loss so far.
        - best_train_epoch: The epoch where the best training loss is achieved.
        - best_valid_epoch: The epoch where the best validation loss is achieved.
        - batch_idx_train: The batch index of the current batch.
        - log_interval: Log training stats every `log_interval` batches.
        - reset_interval: Reset the stats every `reset_interval` batches.
        - valid_interval: Run validation every `valid_interval` batches.
        - env_info: The environment information.
    """
    params = AttributeDict(
        {
            "allowed_excess_duration_ratio": 0.1,
            "subsampling_factor": 2,
            "frame_shift_ms": 10,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 5000,
            "env_info": get_env_info(),
        }
    )

    return params


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""
    texts = []
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
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
    # left padding texts to the same length, texts is a list of list,
    # padding with tokenzier.pad_token_id
    max_len_texts = max([len(text) for text in texts])
    texts = [
        [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text for text in texts
    ]
    input_ids = torch.tensor(texts, dtype=torch.int)

    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    # mask all tokens before token_id 151646 with IGNORE_TOKEN_ID
    # first get the indices of the tokens

    mask_indices = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )
    for i in range(mask_indices[0].size(0)):
        row = mask_indices[0][i]
        col = mask_indices[1][i]
        # + 2 to  skip: 'assistant', '\n'
        target_ids[row, : col + 2] = IGNORE_TOKEN_ID

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return input_ids, attention_mask, target_ids


def compute_loss(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute the loss for the given batch.
    Args:
        params:
            It is returned by :func:`get_params`.
        tokenizer:
            The tokenizer used to encode the text.
        model:
            The model for training.
        batch:
            A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
            for the content in it.
        is_training:
            Whether it is training.
    Returns:
        Return a tuple of two elements. The first element is the loss tensor.
    """

    device = next(model.parameters()).device
    feature = batch["inputs"]

    assert feature.ndim == 3
    feature = feature.to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    texts = batch["supervisions"]["text"]

    messages = []
    for text in texts:
        speech_part = f"{START_SPEECH_TOKEN}{DEFAULT_SPEECH_TOKEN}{END_SPEECH_TOKEN}"
        content_list = [speech_part] * 4
        drop_content_list = [x for x in content_list if random.random() > 1 / 4]
        if len(drop_content_list) == 0:
            drop_content_list = content_list
        content = "".join(drop_content_list)
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": text},
        ]
        messages.append(message)

    input_ids, attention_mask, target_ids = preprocess(messages, tokenizer)

    target_ids = target_ids.type(torch.LongTensor)
    input_ids = input_ids.type(torch.LongTensor)

    with torch.set_grad_enabled(is_training):
        model_outputs, acc = model(
            feature=feature,
            feature_lens=feature_lens,
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=target_ids.to(device),
        )
        loss = model_outputs.loss
    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item() * info["frames"]
    info["acc"] = (
        acc * info["frames"]
    )  # WAR: to avoid normalization by the number of frames

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
            loss, loss_info = compute_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                batch=batch,
                is_training=False,
            )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        if batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        try:
            with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
                loss, loss_info = compute_loss(
                    params=params,
                    tokenizer=tokenizer,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.

            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if params.use_autocast:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it. The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_autocast else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_autocast else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
    exclude_frozen_parameters: bool = True,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
        exclude_frozen_parameters=exclude_frozen_parameters,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if params.use_bf16:  # amp + bf16
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:  # amp + fp16
        params.dtype = torch.float16
        params.use_autocast = True
    else:  # fp32
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(f"Using dtype={params.dtype}")
    logging.info(f"Use AMP={params.use_autocast}")

    logging.info(params)

    logging.info("About to create model")
    speech_encoder_types = params.speech_encoder_type.split(",")
    speech_encoder_paths = params.speech_encoder_path.split(",")
    speech_encoder_bpe_paths = params.speech_encoder_bpe_path.split(",")
    assert len(speech_encoder_paths) == len(speech_encoder_bpe_paths)
    speech_encoder_list = []
    for speech_encoder_type, speech_encoder_path, speech_encoder_bpe_path in zip(
        speech_encoder_types, speech_encoder_paths, speech_encoder_bpe_paths
    ):
        zipformer_model = get_zipformer_model(
            speech_encoder_type, speech_encoder_path, speech_encoder_bpe_path, "cpu"
        )
        speech_encoder = ZipformerEncoderModel(
            zipformer_model.encoder_embed, zipformer_model.encoder
        )
        speech_encoder_list.append(speech_encoder)
    num_param = sum([p.numel() for p in speech_encoder.parameters()])
    logging.info(f"Number of speech encoder parameters: {num_param}")
    speech_encoder = nn.ModuleList(speech_encoder_list)

    if not params.unfreeze_encoder:
        for name, param in speech_encoder.named_parameters():
            param.requires_grad = False
        speech_encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(params.llm_path)

    llm = AutoModelForCausalLM.from_pretrained(
        params.llm_path,
        attn_implementation="flash_attention_2",
        torch_dtype=params.dtype,
    )

    if not params.unfreeze_llm:
        for name, param in llm.named_parameters():
            param.requires_grad = False
        llm.eval()
    else:
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
                lora_dropout=0.05,
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
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )
    llm.config.start_text_token_id = tokenizer.convert_tokens_to_ids(START_TEXT_TOKEN)
    llm.config.end_text_token_id = tokenizer.convert_tokens_to_ids(END_TEXT_TOKEN)
    llm.config.start_speech_token_id = tokenizer.convert_tokens_to_ids(
        START_SPEECH_TOKEN
    )
    llm.config.end_speech_token_id = tokenizer.convert_tokens_to_ids(END_SPEECH_TOKEN)

    encoder_projector_list = []
    for i in range(len(speech_encoder)):
        speech_encoder_dim = max(speech_encoder[i].encoder.encoder_dim)

        encoder_projector = EncoderProjector(
            speech_encoder_dim, llm.config.hidden_size, params.encoder_projector_ds_rate
        )
        encoder_projector_list.append(encoder_projector)
    encoder_projector = nn.ModuleList(encoder_projector_list)

    model = Speech_LLM_Zipformer_MoSE(
        speech_encoder,
        llm,
        encoder_projector,
    )

    if params.pretrained_model_path:
        checkpoint = torch.load(
            params.pretrained_model_path, map_location="cpu", weights_only=False
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    num_trainable_param = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    logging.info(f"Number of trainable model parameters: {num_trainable_param}")

    logging.info("Trainable parameters (excluding model.eval modules):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"{name}: {param.shape}")

    assert params.start_epoch > 0, params.start_epoch

    if params.pretrained_model_path is not None:
        _ = load_checkpoint(filename=params.pretrained_model_path, model=model)
        checkpoints = None
    else:
        checkpoints = load_checkpoint_if_available(params=params, model=model)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    logging.info(f"Device: {device}")

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    # scheduler = Eden(optimizer, warmup_batches=1000)
    scheduler = FixedLRScheduler(optimizer)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.inf_check:
        register_inf_check_hooks(model)

    data_module = AsrDataModule(args)
    multi_dataset = LibriSpeechDataset(args.manifest_dir)

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 20.0:
            # logging.warning(
            #    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    train_cuts = multi_dataset.librispeech_train_cuts()

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    train_dl = data_module.train_dataloaders(train_cuts)

    valid_cuts = multi_dataset.librispeech_dev_cuts()
    valid_dl = data_module.valid_dataloaders(valid_cuts)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    scaler = create_grad_scaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    logging.info(f"start training from epoch {params.start_epoch}")
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=rank,
            exclude_frozen_parameters=True,
        )

    logging.info("Done!")


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size

    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
