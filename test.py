import logging
import os
import subprocess
import tempfile
import unittest

import pytest
from datasets import load_dataset
from executorch import version
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoProcessor, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
from optimum.utils.import_utils import is_transformers_version

model_id = "openai/whisper-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ExecuTorchModelForSpeechSeq2Seq.from_pretrained(model_id, recipe="portable")
processor = AutoProcessor.from_pretrained(model_id)

assert(isinstance(model, ExecuTorchModelForSpeechSeq2Seq))
assert(hasattr(model, "encoder"))
assert(isinstance(model.encoder, ExecuTorchModule))
assert(hasattr(model, "decoder"))
assert(isinstance(model.decoder, ExecuTorchModule))
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

input_features = processor(
            sample["array"], return_tensors="pt", truncation=False, sampling_rate=sample["sampling_rate"]
).input_features
# Current implementation of the transcibe method accepts up to 30 seconds of audio, therefore I trim the audio here.
input_features_trimmed = input_features[:, :, :3000].contiguous()

generated_transcription = model.transcribe(tokenizer, input_features_trimmed)
expected_text = " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins work is really Greek after all, and can discover that."
logging.info(
            f"\nExpected transcription:\n\t{expected_text}\nGenerated transcription:\n\t{generated_transcription}"
)
assert(generated_transcription == expected_text)
