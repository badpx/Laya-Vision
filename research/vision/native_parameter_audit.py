"""Count the proposed native vision architecture on meta, without model downloads.

Run from the repository with its tested environment:
    .venv/bin/python research/vision/native_parameter_audit.py

This is a parameter audit, not an implemented multimodal model or a benchmark.
If the local Laya checkpoint exists, its safetensors header is compared with the
meta model; tensor payloads are never read. All new modules have meta storage.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def checkpoint_header_counts(path):
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_size <= 100_000_000:
            raise ValueError('Invalid or unexpectedly large safetensors header')
        header = json.loads(stream.read(header_size))
    groups = Counter()
    for name, value in header.items():
        if name != '__metadata__':
            groups[name.split('.')[0]] += math.prod(value['shape'])
    return dict(sorted(groups.items()))


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def audit(checkpoint):
    import torch
    from torch import nn
    import transformers
    from transformers import ModernBertConfig, ModernBertModel, SiglipVisionConfig, SiglipVisionModel
    from laya.common import DecisionModel

    # Count-relevant fields of the local multilingual encoder/config.json.
    text_config = ModernBertConfig(
        vocab_size=256000, hidden_size=768, intermediate_size=1152,
        num_hidden_layers=22, num_attention_heads=12, max_position_embeddings=8192,
        attention_bias=False, mlp_bias=False, norm_bias=False,
        pad_token_id=0, cls_token_id=1, sep_token_id=1,
    )
    # The official FixRes384 config uses SiglipVisionModel, not the NaFlex class.
    vision_config = SiglipVisionConfig(
        hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
        num_attention_heads=12, image_size=384, patch_size=16, num_channels=3,
    )

    class CrossAttentionForCounting(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(768)
            self.attention = nn.MultiheadAttention(768, 12, bias=True, batch_first=True)
            self.gate = nn.Parameter(torch.zeros(()))

    with torch.device('meta'):
        laya = DecisionModel(ModernBertModel(text_config), head_layers=2, n_act=2)
        vision = SiglipVisionModel(vision_config)
        projector = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768),
        )
        fusion = nn.ModuleList([CrossAttentionForCounting(), CrossAttentionForCounting()])

    modules = [laya, vision, projector, fusion]
    assert all(parameter.is_meta for module in modules for parameter in module.parameters())
    old_groups = dict(sorted(Counter({
        name: parameter_count(getattr(laya, name))
        for name in ('encoder', 'head', 'scorer', 'act_head', 'type_emb')
    }).items()))
    persistent_buffers = {
        name: tensor.numel() for name, tensor in laya.state_dict().items()
        if name not in dict(laya.named_parameters())
    }
    # In transformers 5.17 this is .head; older releases may nest the tower.
    tower = getattr(vision, 'vision_model', vision)
    pooling = parameter_count(tower.head)
    vision_all = parameter_count(vision)
    vision_dense = vision_all - pooling
    projector_parameters = parameter_count(projector)
    cross_parameters = parameter_count(fusion)
    new_parameters = projector_parameters + cross_parameters
    old_parameters = parameter_count(laya)
    total_parameters = old_parameters + vision_dense + new_parameters
    buffer_count = sum(persistent_buffers.values())

    expected = {
        'old_laya_parameters': 321_908_995,
        'old_laya_persistent_buffer_elements': 3,
        'vision_with_pooling_parameters': 93_176_064,
        'removed_pooling_parameters': 7_087_104,
        'vision_dense_tower_parameters': 86_088_960,
        'projector_parameters': 1_182_720,
        'two_cross_attention_parameters': 4_727_810,
        'new_trainable_parameters': 5_910_530,
        'total_parameters': 413_908_485,
        'total_parameters_plus_persistent_buffers': 413_908_488,
    }
    actual = dict(zip(expected, (
        old_parameters, buffer_count, vision_all, pooling, vision_dense,
        projector_parameters, cross_parameters, new_parameters,
        total_parameters, total_parameters + buffer_count,
    )))
    assert actual == expected, {'expected': expected, 'actual': actual}
    assert projector_parameters == 2 * 768 + 2 * (768 * 768 + 768)
    assert cross_parameters == 2 * (4 * 768 * 768 + 4 * 768 + 2 * 768 + 1)

    header_audit = {'status': 'not_present', 'tensor_payload_read': False}
    if checkpoint.is_file():
        groups = checkpoint_header_counts(checkpoint)
        assert groups == dict(sorted({**old_groups, **persistent_buffers}.items()))
        header_audit.update(status='matched', tensor_elements_by_prefix=groups)

    stage_two = new_parameters + old_groups['head'] + old_groups['scorer']
    return {
        'scope': 'Exact meta parameter counts; no pretrained vision weights, forward, training or benchmark.',
        'environment': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'architecture': {
            'text': 'Existing multilingual Laya: mmBERT-base plus original decision modules',
            'vision': 'google/siglip2-base-patch16-384 dense tower; MAP pooling head excluded',
            'vision_config_source': 'https://huggingface.co/google/siglip2-base-patch16-384/blob/main/config.json',
            'vision_tokens': 576,
            'hidden_size': 768,
            'projector': 'LayerNorm(768, affine) -> Linear(768,768,bias) -> GELU -> Linear(768,768,bias)',
            'fusion': 'Two blocks, each full-width 12-head MHA with biases, one affine LayerNorm and scalar gate; no FFN',
            'excluded_additions': 'No extra trainable modality/position embeddings, tile embeddings, LoRA or auxiliary heads',
        },
        'counts': actual,
        'old_laya_parameter_groups': old_groups,
        'old_laya_persistent_buffers': persistent_buffers,
        'local_checkpoint_header_audit': header_audit,
        'trainable_stages': {
            'stage_1_projector_and_cross_attention': new_parameters,
            'stage_2_add_original_head_and_scorer': stage_two,
            'stage_2_frozen_original_modules': ['encoder', 'type_emb', 'act_head'],
            'act_head_policy': 'Remains frozen unless an explicit supervised action objective is added.',
        },
        'checks': {
            'all_model_parameters_on_meta': True,
            'counts_match_explicit_expected_values': True,
            'projector_formula_matches': True,
            'cross_attention_formula_matches': True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'downloads/vision-models/laya/multilingual/model.safetensors')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/vision/native-parameter-audit.json')
    args = parser.parse_args()
    report = audit(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['counts'], indent=2))
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
