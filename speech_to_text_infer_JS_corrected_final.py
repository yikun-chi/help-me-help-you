# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

from argparse import ArgumentParser

import torch
import json

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
    parser.add_argument(
        "--dont_normalize_text",
        default=False,
        action='store_true',
        help="Turn off trasnscript normalization. Recommended for non-English.",
    )
    parser.add_argument(
        "--use_cer", default=False, action='store_true', help="Use Character Error Rate as the evaluation metric"
    )
    #ADDED
    #Provide a manifest as the reference vocab to compare our examples against
    parser.add_argument(
        "--ref_vocab", type=str, required=True
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)
    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model.decoder.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': not args.dont_normalize_text,
        }
    )
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
    wer = WER(vocabulary=asr_model.decoder.vocabulary,log_prediction=True)
    hypotheses = []
    references = []
    for test_batch in asr_model.test_dataloader():
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        hypotheses += wer.ctc_decoder_predictions_tensor(greedy_predictions)
        for batch_ind in range(greedy_predictions.shape[0]):
            seq_len = test_batch[3][batch_ind].cpu().detach().numpy()
            seq_ids = test_batch[2][batch_ind].cpu().detach().numpy()
            reference = ''.join([labels_map[c] for c in seq_ids[0:seq_len]])
            references.append(reference)
        del test_batch
    ref_vocab_set = set()
    with open(args.ref_vocab, 'r') as f_hot:
      for line in f_hot:
        json_line = json.loads(line)
        line_vocab = json_line['text']
        ref_vocab_set.update(line_vocab.split())

    #ADDED
    # For each example, we split it into words and see if the word is in our reference vocab
    # If ALL words in the example are in the reference vocab, we add both the hypotheses and references
    # to the vocab list
    # Otherwise, if a word was not in the training data, we add it to the OOV list
    hypotheses_vocab = []
    references_vocab = []
    hypotheses_OOV = []
    references_OOV = []
    for i in range(len(references)):
      reference_words=references[i].split()
      ref_num = 0
      ref_denom = 0
      for reference_word in reference_words:
        if reference_word in ref_vocab_set:
          ref_num+=1
        ref_denom=len(reference_words)
      if ref_num==ref_denom:
        references_vocab.append(references[i])
        hypotheses_vocab.append(hypotheses[i])
      else:
        hypotheses_OOV.append(hypotheses[i])
        references_OOV.append(references[i])

    # Now we can calculate and print separate Hot and Cold WERs    
    wer_value_total = word_error_rate(hypotheses=hypotheses, references=references, use_cer=args.use_cer)
    if len(references_vocab) > 0:
      wer_value_vocab = word_error_rate(hypotheses=hypotheses_vocab, references=references_vocab, use_cer=args.use_cer)
    if len(references_OOV) >0:
      wer_value_OOV = word_error_rate(hypotheses=hypotheses_OOV, references=references_OOV, use_cer=args.use_cer)
    if not args.use_cer:
        if wer_value_total > args.wer_tolerance:
            raise ValueError(f"got wer of {wer_value}. it was higher than {args.wer_tolerance}")
        logging.info(f'Got WER of {wer_value_total}. Tolerance was {args.wer_tolerance}')
        if len(references_vocab) > 0:
          logging.info(f'Got hot text WER of {wer_value_vocab}.')
        if len(references_OOV) >0:
          logging.info(f'Got cold text WER of {wer_value_OOV}.')
        
    else:
        logging.info(f'Got CER of {wer_value}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
