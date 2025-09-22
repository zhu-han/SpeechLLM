from typing import Tuple

import sentencepiece as spm
import torch
import torch.nn as nn

from utils import AttributeDict
from utils import save_checkpoint as save_checkpoint_impl
from zipformer.attention_decoder import AttentionDecoderModel
from zipformer.decoder import Decoder
from zipformer.encoder_interface import EncoderInterface
from zipformer.joiner import Joiner
from zipformer.model import AsrModel
from zipformer.scaling import ScheduledFloat
from zipformer.subsampling import Conv2dSubsampling
from zipformer.zipformer import Zipformer2


def get_model_params(type: str = "medium") -> AttributeDict:
    """Return a dict containing Zipformer model parameters."""
    medium_params = AttributeDict(
        {
            "feature_dim": 80,
            "num_encoder_layers": "2,2,3,4,3,2",
            "downsampling_factor": "1,2,4,8,4,2",
            "feedforward_dim": "512,768,1024,1536,1024,768",
            "num_heads": "4,4,4,8,4,4",
            "encoder_dim": "192,256,384,512,384,256",
            "query_head_dim": "32",
            "value_head_dim": "12",
            "pos_head_dim": "4",
            "pos_dim": 48,
            "encoder_unmasked_dim": "192,192,256,256,256,192",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "decoder_dim": 512,
            "joiner_dim": 512,
            "attention_decoder_dim": 512,
            "attention_decoder_num_layers": 6,
            "attention_decoder_attention_dim": 512,
            "attention_decoder_num_heads": 8,
            "attention_decoder_feedforward_dim": 2048,
            "causal": False,
            "chunk_size": "16,32,64,-1",
            "left_context_frames": "64,128,256,-1",
            "context_size": 2,
            "use_transducer": True,
            "use_ctc": False,
            "use_attention_decoder": False,
        }
    )

    large_params = AttributeDict(
        {
            "feature_dim": 80,
            "num_encoder_layers": "2,2,4,5,4,2",
            "downsampling_factor": "1,2,4,8,4,2",
            "feedforward_dim": "512,768,1536,2048,1536,768",
            "num_heads": "4,4,4,8,4,4",
            "encoder_dim": "192,256,512,768,512,256",
            "query_head_dim": "32",
            "value_head_dim": "12",
            "pos_head_dim": "4",
            "pos_dim": 48,
            "encoder_unmasked_dim": "192,192,256,320,256,192",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "decoder_dim": 512,
            "joiner_dim": 512,
            "attention_decoder_dim": 512,
            "attention_decoder_num_layers": 6,
            "attention_decoder_attention_dim": 512,
            "attention_decoder_num_heads": 8,
            "attention_decoder_feedforward_dim": 2048,
            "causal": False,
            "chunk_size": "16,32,64,-1",
            "left_context_frames": "64,128,256,-1",
            "context_size": 2,
            "use_transducer": True,
            "use_ctc": False,
            "use_attention_decoder": False,
        }
    )

    if type == "medium":
        return medium_params
    elif type == "large":
        return large_params
    else:
        raise ValueError(f"Unsupported model type: {type}")


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_attention_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = AttentionDecoderModel(
        vocab_size=params.vocab_size,
        decoder_dim=params.attention_decoder_dim,
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=params.attention_decoder_attention_dim,
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_dim,
        memory_dim=max(_to_int_tuple(params.encoder_dim)),
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return decoder


def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    if params.use_transducer:
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    else:
        decoder = None
        joiner = None

    if params.use_attention_decoder:
        attention_decoder = get_attention_decoder_model(params)
    else:
        attention_decoder = None

    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=attention_decoder,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
        use_attention_decoder=params.use_attention_decoder,
    )
    return model


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_zipformer_model(model_type, model_filename, bpe_filename, device, **kwargs):
    zipformer_params = get_model_params(model_type)
    for key, value in kwargs.items():
        if key in zipformer_params.keys():
            zipformer_params[key] = value
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_filename)
    zipformer_params.blank_id = sp.piece_to_id("<blk>")
    zipformer_params.unk_id = sp.piece_to_id("<unk>")
    zipformer_params.vocab_size = sp.get_piece_size()
    model = get_model(zipformer_params)
    model.load_state_dict(
        torch.load(model_filename, map_location=device, weights_only=False)["model"],
        strict=False,
    )
    return model


class ZipformerEncoderModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
    ):
        """A Wrapper for Zipformer encoder.

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
        """
        super().__init__()

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)
