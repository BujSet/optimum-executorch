# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import os
import sys
import unittest

import pytest
import torchao
import transformers
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoConfig, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM

from ..utils import check_causal_lm_output_quality


os.environ["TOKENIZERS_PARALLELISM"] = "false"
is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
is_linux_ci = sys.platform.startswith("linux") and is_ci


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        is_linux_ci
        or parse(transformers.__version__) < parse("4.52.0")
        or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.52.0 and torchao >= 0.11.0. OOM on linux runner.",
    )
    def test_phi4_text_generation_with_custom_sdpa_and_kv_cache_8da4w_8we(self):
        model_id = "microsoft/Phi-4-mini-instruct"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{"qlinear": True, "qembeeding": True},
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="My favourite condiment is ",
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")

        if not is_ci:
            generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

            # Free memory before loading eager for quality check
            del model
            del tokenizer
            gc.collect()

            self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on torchao >= 0.11.0.",
    )
    def test_phi4_text_generation_with_quantized_pte_from_hub(self):
        model_id = "pytorch/Phi-4-mini-instruct-8da4w"
        config = AutoConfig.from_pretrained(model_id)
        # NOTE: To make the model exportable we need to set the rope scaling to default to avoid hitting
        # the data-dependent control flow in _longrope_frequency_update. Alternatively, we can rewrite
        # that function to avoid the data-dependent control flow.
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["type"] = "default"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id, recipe="xnnpack", config=config, file_name="phi4-mini-8da4w.pte"
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="My favourite condiment is ",
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")

        if not is_ci:
            generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

            # Free memory before loading eager for quality check
            del model
            del tokenizer
            gc.collect()

            self.assertTrue(
                check_causal_lm_output_quality(
                    "microsoft/Phi-4-mini-instruct",
                    generated_tokens,
                )
            )

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skip(
        reason="Require cache_position support in executorch runtime. Re-enable when available.",
    )
    def test_phi4_text_generation_with_quantized_ckp(self):
        model_id = "pytorch/Phi-4-mini-instruct-8da4w"
        config = AutoConfig.from_pretrained(model_id)
        # NOTE: To make the model exportable we need to set the rope scaling to default to avoid hitting
        # the data-dependent control flow in _longrope_frequency_update. Alternatively, we can rewrite
        # that function to avoid the data-dependent control flow.
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["type"] = "default"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            config=config,
            export=True,
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        # Using "pytorch/Phi-4-mini-instruct-8da4w" will end up loading a wrong GPT2Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="My favourite condiment is ",
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")

        if not is_ci:
            generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

            # Free memory before loading eager for quality check
            del model
            del tokenizer
            gc.collect()

            self.assertTrue(
                check_causal_lm_output_quality(
                    "microsoft/Phi-4-mini-instruct",
                    generated_tokens,
                )
            )

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    @pytest.mark.skipif(is_ci, reason="Too big for CI runners")
    def test_phi4_text_generation_portable(self):
        model_id = "microsoft/Phi-4-mini-instruct"
        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe="portable")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="My favourite condiment is ",
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")

        if not is_ci:
            generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

            # Free memory before loading eager for quality check
            del model
            del tokenizer
            gc.collect()

            self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))
