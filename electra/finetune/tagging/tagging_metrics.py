# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Metrics for sequence tagging tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import numpy as np

from finetune import scorer
from finetune.tagging import tagging_utils

from datasets import load_metric


class WordLevelScorer(scorer.Scorer):
    """Base class for tagging scorers."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(WordLevelScorer, self).__init__()
        self._total_loss = 0
        self._total_words = 0
        self._labels = []
        self._preds = []

    def update(self, results):
        super(WordLevelScorer, self).update(results)
        self._total_loss += results["loss"]
        n_words = int(round(np.sum(results["labels_mask"])))
        self._labels.append(results["labels"][:n_words])
        self._preds.append(results["predictions"][:n_words])
        self._total_loss += np.sum(results["loss"])
        self._total_words += n_words

    def get_loss(self):
        return self._total_loss / max(1, self._total_words)


class AccuracyScorer(WordLevelScorer):
    """Computes accuracy scores."""

    def __init__(self, auto_fail_label=None):
        super(AccuracyScorer, self).__init__()
        self._auto_fail_label = auto_fail_label

    def _get_results(self):
        correct, count = 0, 0
        for labels, preds in zip(self._labels, self._preds):
            for y_true, y_pred in zip(labels, preds):
                count += 1
                correct += (
                    1 if y_pred == y_true and y_true != self._auto_fail_label else 0
                )
        return [("accuracy", 100.0 * correct / count), ("loss", self.get_loss())]


class F1Scorer(WordLevelScorer):
    """Computes F1 scores."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(F1Scorer, self).__init__()
        self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0

    def _get_results(self):
        if self._n_correct == 0:
            p, r, f1 = 0, 0, 0
        else:
            p = 100.0 * self._n_correct / self._n_predicted
            r = 100.0 * self._n_correct / self._n_gold
            f1 = 2 * p * r / (p + r)
        return [
            ("precision", p),
            ("recall", r),
            ("f1", f1),
            ("loss", self.get_loss()),
        ]


class EntityLevelF1Scorer(F1Scorer):
    """Computes F1 score for entity-level tasks such as NER."""

    def __init__(self, label_mapping):
        super(EntityLevelF1Scorer, self).__init__()
        self._inv_label_mapping = {v: k for k, v in six.iteritems(label_mapping)}

    def _get_results(self):
        self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
        for labels, preds in zip(self._labels, self._preds):
            sent_spans = set(
                tagging_utils.get_span_labels(labels, self._inv_label_mapping)
            )
            span_preds = set(
                tagging_utils.get_span_labels(preds, self._inv_label_mapping)
            )
            self._n_correct += len(sent_spans & span_preds)
            self._n_gold += len(sent_spans)
            self._n_predicted += len(span_preds)
        return super(EntityLevelF1Scorer, self)._get_results()


class EntityLevelF1AccuracyScorer(F1Scorer):
    """Computes F1 score for entity-level tasks such as NER and the accuracy at token level."""

    def __init__(self, label_mapping):
        super(EntityLevelF1AccuracyScorer, self).__init__()
        self._inv_label_mapping = {v: k for k, v in six.iteritems(label_mapping)}

    def _get_results(self):
        self._n_correct, self._n_predicted, self._n_gold = 0, 0, 0
        correct, count = 0, 0
        for labels, preds in zip(self._labels, self._preds):
            if labels.size == 0 or preds.size == 0:
                continue
            # accuracy
            for y_true, y_pred in zip(labels, preds):
                count += 1
                correct += 1 if y_pred == y_true else 0
            # f1
            sent_spans = set(
                tagging_utils.get_span_labels(labels, self._inv_label_mapping)
            )
            span_preds = set(
                tagging_utils.get_span_labels(preds, self._inv_label_mapping)
            )
            self._n_correct += len(sent_spans & span_preds)
            self._n_gold += len(sent_spans)
            self._n_predicted += len(span_preds)
        f1_scores = super(EntityLevelF1AccuracyScorer, self)._get_results()

        return [("accuracy", 100.0 * correct / count)] + f1_scores


class CoNLL2002NERScorer(EntityLevelF1AccuracyScorer):
    """Adds F1 score CoNLL2002 style for NER."""

    def __init__(self, label_mapping):
        super(CoNLL2002NERScorer, self).__init__(label_mapping)
        self._inv_label_mapping = {v: k for k, v in six.iteritems(label_mapping)}
        self._conll_metric = load_metric("seqeval")

    def _get_results(self):
        entity_level_f1 = super(CoNLL2002NERScorer, self)._get_results()
        references = [
            [self._inv_label_mapping[tag_id] for tag_id in tags]
            for tags in self._labels
        ]
        predictions = [
            [self._inv_label_mapping[tag_id] for tag_id in tags] for tags in self._preds
        ]
        metrics = self._conll_metric.compute(
            predictions=predictions, references=references
        )

        return [
            ("conll_acc", 100*metrics["overall_accuracy"]),
            ("conll_f1", 100*metrics["overall_f1"]),
            ("conll_p", 100*metrics["overall_precision"]),
            ("conll_r", 100*metrics["overall_recall"]),
        ] + entity_level_f1
