"""Small, offline regression tests for the notebook's training workflow."""

import gzip
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import altair as alt
import pandas as pd
import torch

import the_annotated_transformer as transformer


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.weight * src


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def scalar_batch(target):
    return SimpleNamespace(
        src=torch.ones(1),
        tgt=torch.zeros(1),
        src_mask=None,
        tgt_mask=None,
        tgt_y=torch.tensor([target], dtype=torch.float32),
        ntokens=torch.tensor(1),
    )


def scalar_loss(out, target, ntokens):
    loss = ((out - target) ** 2).mean() / 2
    return loss.detach() * ntokens, loss


class TrainingWorkflowTests(unittest.TestCase):
    def test_device_selection_can_force_cpu(self):
        with patch.dict(os.environ, {"ANNOTATED_TRANSFORMER_DEVICE": "cpu"}):
            config = transformer.training_config()
        self.assertEqual(config["device"], "cpu")
        with (
            patch.object(
                torch.cuda, "is_available", side_effect=AssertionError
            ),
            patch.object(
                transformer,
                "make_model",
                side_effect=RuntimeError("model stub"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "model stub"):
                transformer.train_worker(
                    0, 1, {"<blank>": 2}, {"<blank>": 2}, None, None, config
                )
        with patch.dict(
            os.environ, {"ANNOTATED_TRANSFORMER_DEVICE": "invalid"}
        ):
            with self.assertRaises(ValueError):
                transformer.training_config()

    def test_accumulation_updates_and_flushes_short_final_window(self):
        model = ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CountingScheduler()

        _, state = transformer.run_epoch(
            [scalar_batch(1), scalar_batch(3), scalar_batch(2)],
            model,
            scalar_loss,
            optimizer,
            scheduler,
            accum_iter=2,
        )

        # Two batches average to gradient -2, then the final gradient is -1.8.
        self.assertAlmostEqual(model.weight.item(), 0.38, places=6)
        self.assertEqual(scheduler.steps, 2)
        self.assertEqual(state.step, 3)
        self.assertEqual(state.accum_step, 2)
        self.assertEqual(state.samples, 3)

    def test_evaluation_does_not_update_model_or_scheduler(self):
        model = ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CountingScheduler()
        _, state = transformer.run_epoch(
            [scalar_batch(1)],
            model,
            scalar_loss,
            optimizer,
            scheduler,
            mode="eval",
        )
        self.assertEqual(model.weight.item(), 0)
        self.assertEqual(scheduler.steps, 0)
        self.assertEqual(state.step, 0)

    def test_collate_truncates_and_keeps_eos_and_padding(self):
        def token_ids(tokens):
            return [int(token) for token in tokens]

        src, tgt = transformer.collate_batch(
            [("3 4 5 6", "8 9 10 11"), ("7", "12")],
            str.split,
            str.split,
            token_ids,
            token_ids,
            torch.device("cpu"),
            max_padding=5,
        )
        self.assertEqual(src.tolist(), [[0, 3, 4, 5, 1], [0, 7, 1, 2, 2]])
        self.assertEqual(tgt.tolist(), [[0, 8, 9, 10, 1], [0, 12, 1, 2, 2]])
        with self.assertRaises(ValueError):
            transformer.collate_batch(
                [("3", "4")],
                str.split,
                str.split,
                token_ids,
                token_ids,
                torch.device("cpu"),
                max_padding=1,
            )

    def test_greedy_decode_stops_after_eos(self):
        class FakeModel:
            def encode(self, src, src_mask):
                return src

            def decode(self, memory, src_mask, ys, tgt_mask):
                return ys.unsqueeze(-1).float()

            def generator(self, last):
                token = 3 if last.item() == 0 else 1
                scores = torch.zeros(1, 4)
                scores[0, token] = 1
                return scores

        decoded = transformer.greedy_decode(
            FakeModel(),
            torch.ones(1, 1, dtype=torch.long),
            None,
            max_len=8,
            start_symbol=0,
            eos_symbol=1,
        )
        self.assertEqual(decoded.tolist(), [[0, 3, 1]])

    def test_average_uses_source_models_without_mutating_them(self):
        target, first, second = [torch.nn.Linear(1, 1) for _ in range(3)]
        with torch.no_grad():
            target.weight.fill_(99)
            target.bias.fill_(99)
            first.weight.fill_(1)
            first.bias.fill_(3)
            second.weight.fill_(5)
            second.bias.fill_(7)
        transformer.average(target, [first, second])
        self.assertEqual(target.weight.item(), 3)
        self.assertEqual(target.bias.item(), 5)
        self.assertEqual(first.weight.item(), 1)
        self.assertEqual(second.weight.item(), 5)


class DatasetTests(unittest.TestCase):
    def write_split(self, root, name, language, text):
        path = (
            root / f"{name}.{language}.{transformer.MULTI30K_REVISION[:12]}.gz"
        )
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)

    def test_unaligned_dataset_is_rejected_even_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_split(root, "train", "de", "eins\nzwei\n")
            self.write_split(root, "train", "en", "one\n")
            with self.assertRaisesRegex(ValueError, "Unaligned"):
                transformer.load_multi30k("train", limit=1, root=root)

    def test_vocabulary_order_and_unknown_tokens(self):
        vocab = transformer.build_vocab_from_iterator(
            [["z", "a", "z", "a", "b"], ["b", "z", "once"]],
            min_freq=2,
            specials=transformer.SPECIALS,
        )
        self.assertEqual(vocab.get_itos(), transformer.SPECIALS + ["z", "a", "b"])
        self.assertEqual(vocab(["z", "missing"]), [4, 3])
        self.assertNotIn("once", vocab)

    def test_vocabulary_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_split(root, "train", "de", "katze\nkatze\n")
            self.write_split(root, "train", "en", "cat\ncat\n")
            de, en = transformer.load_tokenizers()
            first = transformer.load_vocab(de, en, root=root)
            with patch.object(transformer, "build_vocabulary", side_effect=AssertionError):
                second = transformer.load_vocab(de, en, root=root)
            self.assertEqual([v.get_itos() for v in first], [v.get_itos() for v in second])
            self.assertEqual(second[0]["unknown"], second[0]["<unk>"])

    def test_vocabulary_only_uses_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_split(root, "train", "de", "katze\nkatze\n")
            self.write_split(root, "train", "en", "cat\ncat\n")
            self.write_split(root, "val", "de", "hund\nhund\n")
            self.write_split(root, "val", "en", "dog\ndog\n")
            de, en = transformer.load_tokenizers()
            vocab_de, vocab_en = transformer.build_vocabulary(
                de, en, root=root
            )
            self.assertIn("katze", vocab_de)
            self.assertIn("cat", vocab_en)
            self.assertNotIn("hund", vocab_de)
            self.assertNotIn("dog", vocab_en)


class VisualizationTests(unittest.TestCase):
    def test_nested_attention_charts_can_be_serialized(self):
        chart = (
            alt.Chart(pd.DataFrame({"x": [0], "y": [1]}))
            .mark_rect()
            .encode(x="x:O", y="y:O")
        )
        combined = alt.hconcat(
            alt.vconcat(chart | chart) & alt.vconcat(chart | chart)
        )
        self.assertIn("hconcat", combined.to_dict())


if __name__ == "__main__":
    unittest.main()
