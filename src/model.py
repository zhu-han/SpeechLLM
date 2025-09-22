import random

import torch
from torch import nn
from transformers.trainer_pt_utils import LabelSmoother
from zipformer.scaling import SwooshR

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class Speech_LLM_Zipformer(nn.Module):
    """
    The Speech-to-Text model. It consists of an encoder, a language model and an encoder projector.
    The encoder is used to extract speech features from the input speech signal.
    The encoder projector is used to project the encoder outputs to the same dimension as the language model.
    The language model is used to generate the text from the speech features.
    Args:
        encoder (:obj:`nn.Module`): The encoder module.
        llm (:obj:`nn.Module`): The language model module.
        encoder_projector (:obj:`nn.Module`): The encoder projector module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        num_prompt: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.prompt_embedding = torch.nn.Embedding(num_prompt, llm.config.hidden_size)

    def _merge_input_ids_with_speech_features(
        self,
        speech_features,
        speech_feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels=None,
    ):
        """
        Merge the speech features with the input_ids and attention_mask. This is done by replacing the speech tokens
        with the speech features and padding the input_ids to the maximum length of the speech features.
        Args:
            speech_features (:obj:`torch.Tensor`): The speech features to merge with the input_ids.
            speech_feature_lens (:obj:`torch.Tensor`): The lengths of the speech features.
            inputs_embeds (:obj:`torch.Tensor`): The embeddings of the input_ids.
            input_ids (:obj:`torch.Tensor`): The input ids to merge.
            attention_mask (:obj:`torch.Tensor`): The attention mask to merge.
            labels (:obj:`torch.Tensor`, `optional`): The labels to merge.
        Returns:
            :obj:`Tuple(torch.Tensor)`: The merged embeddings, attention mask, labels and position ids.
        """
        batch_size = speech_features.shape[0]
        final_inputs_embeds = []
        final_attention_masks = []
        final_outputs_labels = []
        for i in range(batch_size):
            start_text_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.start_text_token_id
            )[0]
            end_text_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.end_text_token_id
            )[0]
            start_speech_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.start_speech_token_id
            )[0]
            end_speech_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.end_speech_token_id
            )[0]
            speech_id_pos = torch.where(
                input_ids[i][attention_mask[i]]
                == self.llm.config.default_speech_token_id
            )[0]
            assert speech_id_pos.size(0) == 1, f"speech_id_pos: {speech_id_pos}"
            if start_text_id_pos.size(0) > 0:
                inputs_embeds[i][attention_mask[i]][
                    start_text_id_pos[0]
                ] = self.prompt_embedding.weight[0].to(inputs_embeds.dtype)
            if end_text_id_pos.size(0) > 0:
                inputs_embeds[i][attention_mask[i]][
                    end_text_id_pos[0]
                ] = self.prompt_embedding.weight[1].to(inputs_embeds.dtype)
            if start_speech_id_pos.size(0) > 0:
                inputs_embeds[i][attention_mask[i]][
                    start_speech_id_pos[0]
                ] = self.prompt_embedding.weight[2].to(inputs_embeds.dtype)
            if end_speech_id_pos.size(0) > 0:
                inputs_embeds[i][attention_mask[i]][
                    end_speech_id_pos[0]
                ] = self.prompt_embedding.weight[3].to(inputs_embeds.dtype)
            final_inputs_embed = torch.cat(
                (
                    inputs_embeds[i][attention_mask[i]][:speech_id_pos],
                    speech_features[i][: speech_feature_lens[i]],
                    inputs_embeds[i][attention_mask[i]][speech_id_pos + 1 :],
                ),
                dim=0,
            )
            final_attention_mask = torch.full(
                (int(speech_feature_lens[i]) + attention_mask[i].sum() - 1,),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            final_inputs_embeds.append(final_inputs_embed.flip(dims=[0]))
            final_attention_masks.append(final_attention_mask.flip(dims=[0]))

            if labels is not None:
                final_outputs_label = torch.cat(
                    (
                        labels[i][attention_mask[i]][:speech_id_pos],
                        torch.full(
                            (int(speech_feature_lens[i]),),
                            IGNORE_TOKEN_ID,
                            dtype=labels.dtype,
                            device=labels.device,
                        ),
                        labels[i][attention_mask[i]][speech_id_pos + 1 :],
                    ),
                    dim=0,
                )
                final_outputs_labels.append(final_outputs_label.flip(dims=[0]))

        final_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            final_inputs_embeds, batch_first=True, padding_value=0.0
        ).flip(dims=[1])
        final_attention_masks = torch.nn.utils.rnn.pad_sequence(
            final_attention_masks, batch_first=True, padding_value=False
        ).flip(dims=[1])
        assert final_attention_masks.size(1) == final_inputs_embeds.size(
            1
        ), f"{final_attention_masks.size()}, {final_inputs_embeds.size()}"

        if labels is not None:
            final_outputs_labels = torch.nn.utils.rnn.pad_sequence(
                final_outputs_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID
            ).flip(dims=[1])
            assert final_outputs_labels.size(1) == final_inputs_embeds.size(
                1
            ), f"{final_outputs_labels.size()}, {final_inputs_embeds.size()}"
        else:
            final_outputs_labels = None

        return final_inputs_embeds, final_attention_masks, final_outputs_labels

    def forward(
        self,
        feature: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
    ):
        with torch.set_grad_enabled(self.encoder.training):
            encoder_outs, encoder_lens = self.encoder(feature, feature_lens)
        speech_features = self.encoder_projector(encoder_outs)
        encoder_lens = encoder_lens // self.encoder_projector.downsample_rate
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        (
            inputs_embeds,
            attention_mask,
            labels,
        ) = self._merge_input_ids_with_speech_features(
            speech_features,
            encoder_lens,
            inputs_embeds,
            input_ids,
            attention_mask,
            labels,
        )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return model_outputs, acc

    def decode(
        self,
        feature: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):

        encoder_outs, encoder_lens = self.encoder(feature, feature_lens)
        speech_features = self.encoder_projector(encoder_outs)
        encoder_lens = encoder_lens // self.encoder_projector.downsample_rate
        speech_features = speech_features.to(torch.float16)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            _,
        ) = self._merge_input_ids_with_speech_features(
            speech_features, encoder_lens, inputs_embeds, input_ids, attention_mask
        )
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids


class Speech_LLM_Zipformer_MoSE(nn.Module):
    def __init__(
        self,
        encoder: nn.ModuleList,
        llm: nn.Module,
        encoder_projector: nn.ModuleList,
        num_prompt: int = 4,
    ):
        assert len(encoder) == len(
            encoder_projector
        ), f"{len(encoder)}, {len(encoder_projector)}"
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.prompt_embedding = torch.nn.Embedding(num_prompt, llm.config.hidden_size)

    def _merge_input_ids_with_speech_features(
        self,
        speech_features_list,
        speech_feature_lens_list,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels=None,
    ):
        """
        Merge the speech features with the input_ids and attention_mask. This is done by replacing the speech tokens
        with the speech features and padding the input_ids to the maximum length of the speech features.
        Args:
            speech_features (:obj:`torch.Tensor`): The speech features to merge with the input_ids.
            inputs_embeds (:obj:`torch.Tensor`): The embeddings of the input_ids.
            input_ids (:obj:`torch.Tensor`): The input ids to merge.
            attention_mask (:obj:`torch.Tensor`): The attention mask to merge.
            labels (:obj:`torch.Tensor`, `optional`): The labels to merge.
        Returns:
            :obj:`Tuple(torch.Tensor)`: The merged embeddings, attention mask, labels and position ids.
        """
        batch_size = speech_features_list[0].size(0)
        final_inputs_embeds = []
        final_attention_masks = []
        final_outputs_labels = []
        for i in range(batch_size):
            start_text_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.start_text_token_id
            )[0]
            end_text_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.end_text_token_id
            )[0]
            start_speech_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.start_speech_token_id
            )[0]
            end_speech_id_pos = torch.where(
                input_ids[i][attention_mask[i]] == self.llm.config.end_speech_token_id
            )[0]
            speech_id_pos = torch.where(
                input_ids[i][attention_mask[i]]
                == self.llm.config.default_speech_token_id
            )[0]

            special_tokens = {
                start_text_id_pos: 0,
                end_text_id_pos: 1,
                start_speech_id_pos: 2,
                end_speech_id_pos: 3,
            }
            for pos, weight_idx in special_tokens.items():
                if pos.size(0) > 0:
                    inputs_embeds[i][attention_mask[i]][
                        pos[0]
                    ] = self.prompt_embedding.weight[weight_idx].to(inputs_embeds.dtype)

            n = speech_id_pos.size(0)
            sorted_speech_pos = torch.sort(speech_id_pos)[0]

            if n == len(speech_features_list):
                selected_features = speech_features_list
                selected_lens = speech_feature_lens_list
            else:
                available_indices = list(range(len(speech_features_list)))
                selected_indices = random.sample(available_indices, n)
                selected_features = [speech_features_list[j] for j in selected_indices]
                selected_lens = [speech_feature_lens_list[j] for j in selected_indices]

            current_start = 0
            parts = []
            for idx in range(n):
                pos = sorted_speech_pos[idx].item()
                parts.append(inputs_embeds[i][attention_mask[i]][current_start:pos])
                parts.append(selected_features[idx][i][: selected_lens[idx][i]])
                current_start = pos + 1
            parts.append(inputs_embeds[i][attention_mask[i]][current_start:])
            final_inputs_embed = torch.cat(parts, dim=0)

            total_feature_len = sum(selected_lens[idx][i].item() for idx in range(n))
            total_length = total_feature_len + (attention_mask[i].sum().item() - n)
            final_attention_mask = torch.full(
                (total_length,),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            final_outputs_label = None
            if labels is not None:
                current_start = 0
                label_parts = []
                for idx in range(n):
                    pos = sorted_speech_pos[idx].item()
                    label_parts.append(labels[i][attention_mask[i]][current_start:pos])
                    label_parts.append(
                        torch.full(
                            (selected_lens[idx][i].item(),),
                            IGNORE_TOKEN_ID,
                            dtype=labels.dtype,
                            device=labels.device,
                        )
                    )
                    current_start = pos + 1
                label_parts.append(labels[i][attention_mask[i]][current_start:])
                final_outputs_label = torch.cat(label_parts, dim=0)
            final_inputs_embeds.append(final_inputs_embed.flip(dims=[0]))
            final_attention_masks.append(final_attention_mask.flip(dims=[0]))
            if labels is not None:
                final_outputs_labels.append(final_outputs_label.flip(dims=[0]))
        final_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            final_inputs_embeds, batch_first=True, padding_value=0
        ).flip(dims=[1])
        final_attention_masks = torch.nn.utils.rnn.pad_sequence(
            final_attention_masks, batch_first=True, padding_value=False
        ).flip(dims=[1])
        final_outputs_labels = (
            torch.nn.utils.rnn.pad_sequence(
                final_outputs_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID
            ).flip(dims=[1])
            if labels is not None
            else None
        )

        return final_inputs_embeds, final_attention_masks, final_outputs_labels

    def forward(
        self,
        feature: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
    ):
        speech_features_list = []
        for i in range(len(self.encoder)):
            with torch.no_grad():
                encoder_outs, encoder_lens = self.encoder[i](feature, feature_lens)
            speech_features = self.encoder_projector[i](encoder_outs)
            speech_features_list.append(speech_features)
        speech_feature_lens_list = [
            encoder_lens // self.encoder_projector[i].downsample_rate
            for i in range(len(self.encoder))
        ]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            labels,
        ) = self._merge_input_ids_with_speech_features(
            speech_features_list=speech_features_list,
            speech_feature_lens_list=speech_feature_lens_list,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return model_outputs, acc

    def decode(
        self,
        feature: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):

        speech_features_list = []
        for i in range(len(self.encoder)):
            with torch.no_grad():
                encoder_outs, encoder_lens = self.encoder[i](feature, feature_lens)
            speech_features = self.encoder_projector[i](encoder_outs)
            speech_features = speech_features.to(torch.float16)
            speech_features_list.append(speech_features)
        speech_feature_lens_list = [
            encoder_lens // self.encoder_projector[i].downsample_rate
            for i in range(len(self.encoder))
        ]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            _,
        ) = self._merge_input_ids_with_speech_features(
            speech_features_list=speech_features_list,
            speech_feature_lens_list=speech_feature_lens_list,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids


class EncoderProjector(nn.Module):
    """
    The encoder projector module. It is used to project the encoder outputs to the same dimension as the language model.
    Modified from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.
    Args:
        encoder_dim (:obj:`int`): The dimension of the encoder outputs.
        llm_dim (:obj:`int`): The dimension of the language model.
        downsample_rate (:obj:`int`, `optional`, defaults to 5): The downsample rate to use.
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate=5):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim * self.downsample_rate, llm_dim),
            SwooshR(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):

        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )

        x = self.proj(x)
        return x


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.
    Copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/utils/metric.py
    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()
