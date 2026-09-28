"""Synthetic mechanics audit: checkpoint/restart, role guard, masks and gradients.

These tests do not measure genomic accuracy. They run small CPU models and use
only temporary synthetic arrays. A subprocess interruption immediately after an
atomic checkpoint models preemption without modifying train.py.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from model import NeuralCaller, ModelConfig, MODEL_VERSION, tensors, load_model
import train


def synthetic_features(n=8, seed=317):
    rng = np.random.default_rng(seed)
    counts = 2 + np.arange(n) % 2
    am = np.arange(12)[None, :] < counts[:, None]
    gm = np.arange(78)[None, :] < (counts * (counts + 1) // 2)[:, None]
    gp = rng.random((n, 78)) * gm
    gp /= gp.sum(axis=1, keepdims=True)
    # Distinct artificial read likelihoods, not copied from GP or synthetic labels.
    pl = rng.uniform(0, 60, (n, 78)).astype(np.float32)
    pl = (pl - np.where(gm, pl, np.inf).min(1, keepdims=True)) * gm
    fragment_mask = np.arange(4)[None, :] < (np.arange(n) % 5)[:, None]
    fragment = rng.normal(size=(n, 4, 20)).astype(np.float32)
    fragment[:, :, :12] = -np.abs(fragment[:, :, :12]) * 3
    fragment[:, :, :12] *= am[:, None, :]
    fragment *= fragment_mask[:, :, None]
    seq = rng.integers(1, 5, size=(n, 12, 32), dtype=np.int8) * am[:, :, None]
    context = rng.random((n, 12, 6)).astype(np.float32) * am[:, :, None]
    return dict(allele_sequences=seq, allele_mask=am, genotype_mask=gm,
                fragment_features=fragment, fragment_mask=fragment_mask,
                allele_features=rng.random((n, 12, 8)).astype(np.float32) * am[:, :, None],
                depth=(np.arange(n) % 5).astype(np.float32), base_gp=gp.astype(np.float32),
                read_pl=pl, cohort_context=context,
                context_strength=np.full(n, .4, dtype=np.float32))


def assert_tree_equal(test, first, second, prefix='root'):
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second), f'Tensor mismatch: {prefix}')
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second), prefix)
        for key in first: assert_tree_equal(test, first[key], second[key], prefix + '.' + str(key))
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second), prefix)
        for i, (left, right) in enumerate(zip(first, second)):
            assert_tree_equal(test, left, right, prefix + f'[{i}]')
    else:
        test.assertEqual(first, second, prefix)


class ModelMechanicsAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(811)
        model = NeuralCaller(ModelConfig(hidden=8, dropout=0, prior_dropout=0, cohort_dropout=0))
        # Activate all paths: the zero-initialized output otherwise hides masking defects.
        torch.nn.init.normal_(model.pair_joint[-1].weight, std=.15)
        return model.eval()

    def test_padding_never_changes_valid_logits_with_active_head(self):
        model = self.model()
        batch = synthetic_features()
        original = model(tensors(batch, 'cpu')).detach()
        changed = {k: v.copy() for k, v in batch.items()}
        changed['allele_sequences'][~batch['allele_mask']] = 4
        changed['allele_features'][~batch['allele_mask']] = 100
        changed['cohort_context'][~batch['allele_mask']] = 100
        changed['fragment_features'][~batch['fragment_mask']] = 100
        changed['fragment_features'][:, :, :12] = np.where(
            batch['allele_mask'][:, None, :], changed['fragment_features'][:, :, :12], -100)
        changed['base_gp'][~batch['genotype_mask']] = 100
        changed['read_pl'][~batch['genotype_mask']] = 999
        actual = model(tensors(changed, 'cpu')).detach()
        self.assertTrue(torch.equal(original, actual))
        self.assertTrue(torch.all(actual[~torch.from_numpy(batch['genotype_mask'])] == -1e4))

    def test_alternate_allele_permutation_preserves_genotype_meaning(self):
        model = self.model()
        data = {k: v[1:2].copy() for k, v in synthetic_features().items()}
        permutation = np.array([0, 2, 1] + list(range(3, 12)))
        index = []
        for b in range(12):
            for a in range(b + 1):
                aa, bb = sorted((permutation[a], permutation[b]))
                index.append(bb * (bb + 1) // 2 + aa)
        changed = {k: v.copy() for k, v in data.items()}
        for key in ['allele_sequences', 'allele_mask', 'allele_features', 'cohort_context']:
            changed[key] = data[key][:, permutation]
        changed['fragment_features'][:, :, :12] = data['fragment_features'][:, :, permutation]
        for key in ['base_gp', 'read_pl', 'genotype_mask']:
            changed[key] = data[key][:, index]
        expected = model(tensors(data, 'cpu')).detach()[:, index]
        actual = model(tensors(changed, 'cpu')).detach()
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-6)

    def test_nonempty_context_has_gradient_after_zero_output_warmup(self):
        torch.manual_seed(773)
        model = NeuralCaller(ModelConfig(hidden=8, dropout=0, prior_dropout=0, cohort_dropout=0))
        batch = tensors(synthetic_features(), 'cpu')
        labels = torch.tensor([0, 1, 2, 3, 0, 4, 1, 5])
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch, return_aux=True)
            loss = torch.nn.functional.cross_entropy(result['logits'], labels)
            loss.backward()
            gradients = {name: 0 if value.grad is None else float(value.grad.abs().sum())
                         for name, value in model.named_parameters()}
            optimizer.step()
        self.assertGreater(sum(v for k, v in gradients.items() if k.startswith('cohort.')), 1e-12)
        self.assertGreater(sum(v for k, v in gradients.items() if k.startswith('context_gate.')), 1e-12)

    def test_empty_context_exactly_equals_disabled_path(self):
        model = self.model()
        data = synthetic_features()
        data['context_strength'][:] = 0
        batch = tensors(data, 'cpu')
        enabled = model(batch, return_aux=True)
        disabled = model(batch, return_aux=True, cohort_enabled=False)
        self.assertTrue(torch.equal(enabled['logits'], disabled['logits']))
        self.assertTrue(torch.equal(enabled['read_logits'], disabled['read_logits']))
        self.assertTrue(torch.all(enabled['cohort_gate'] == 0))

    def test_auxiliary_read_head_does_not_consume_gp_or_panel_af(self):
        model = self.model()
        data = synthetic_features()
        before = model(tensors(data, 'cpu'), return_aux=True)
        changed = {k: v.copy() for k, v in data.items()}
        changed['base_gp'][:] = 0
        changed['base_gp'][:, 0] = 1
        changed['allele_features'][:, :, 1] = 0  # panel AF
        changed['allele_features'][:, :, 5] = 0  # known AF/panel indicator
        after = model(tensors(changed, 'cpu'), return_aux=True)
        self.assertTrue(torch.equal(before['read_logits'], after['read_logits']))
        self.assertFalse(torch.equal(before['logits'], after['logits']))

    def test_untrained_anchor_uses_gp_not_read_pl(self):
        torch.manual_seed(843)
        model = NeuralCaller(ModelConfig(hidden=8, dropout=0, prior_dropout=0, cohort_dropout=0)).eval()
        data = synthetic_features()
        output = torch.softmax(model(tensors(data, 'cpu')), 1).detach().numpy()
        np.testing.assert_allclose(output, data['base_gp'], rtol=2e-6, atol=1e-7)
        self.assertTrue(np.all(output[~data['genotype_mask']] == 0))


class TrainingRestartAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='neural-restart-audit-')
        self.root = Path(self.directory.name)
        # Two entries exercise both entry cursor and within-entry batch cursor.
        entries = []
        for part in range(2):
            feature = self.root / f'features{part}.npz'
            label = self.root / f'labels{part}.npz'
            values = synthetic_features(n=8, seed=351 + part)
            np.savez_compressed(feature, **values)
            np.savez_compressed(label, labels=np.array([0, 1, 2, 3, 0, 4, 1, -1]))
            entries.append(dict(features=feature.name, labels=label.name, samples=['SYNTHETIC']))
        for role in ['train', 'validation', 'test']:
            (self.root / (role + '.json')).write_text(json.dumps(
                dict(role=role, entries=entries, samples=['SYNTHETIC'],
                     split={'purpose': 'synthetic mechanics only, not genomic validation'})))

    def tearDown(self): self.directory.cleanup()

    def arguments(self, out, *, epochs=2, max_steps=None):
        args = ['--train', str(self.root / 'train.json'), '--validation', str(self.root / 'validation.json'),
                '--out', str(out), '--epochs', str(epochs), '--batch-size', '2', '--hidden', '8',
                '--device', os.environ.get('NEURAL_AUDIT_DEVICE', 'cpu'), '--threads', '1', '--checkpoint-steps', '1', '--seed', '732']
        if max_steps is not None: args += ['--max-steps', str(max_steps)]
        return args

    def run_training(self, out, *, resume=False, stop_after_checkpoint=None,
                     epochs=2, max_steps=None, success=True):
        args = self.arguments(out, epochs=epochs, max_steps=max_steps)
        if resume: args += ['--resume']
        if stop_after_checkpoint is None:
            command = [sys.executable, str(HERE / 'train.py'), *args]
        else:
            program = (
                'import sys;sys.path.insert(0,' + repr(str(HERE)) + ');import train\n'
                'original=train.atomic_save\n'
                'def stop_after_atomic_checkpoint(obj,path):\n'
                ' original(obj,path)\n'
                ' if str(path).endswith("last.pt") and obj["step"]==' + str(stop_after_checkpoint) + ':\n'
                '  raise SystemExit(73)\n'
                'train.atomic_save=stop_after_atomic_checkpoint\n'
                'sys.argv=["train.py"]+' + repr(args) + '\ntrain.main()\n')
            command = [sys.executable, '-c', program]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-5000:])
        elif stop_after_checkpoint is not None:
            self.assertEqual(result.returncode, 73, result.stdout[-2000:] + result.stderr[-5000:])
        else:
            self.assertNotEqual(result.returncode, 0, 'Invalid run unexpectedly succeeded')
        return result

    def load(self, out, name='last.pt'):
        return torch.load(Path(out) / name, map_location='cpu', weights_only=True)

    def test_atomic_checkpoint_restart_matches_uninterrupted_training(self):
        full, interrupted = self.root / 'full', self.root / 'interrupted'
        self.run_training(full)
        self.run_training(interrupted, stop_after_checkpoint=3, success=False)
        partial = self.load(interrupted)
        self.assertEqual(partial['step'], 3)
        self.run_training(interrupted, resume=True)
        first, second = self.load(full), self.load(interrupted)
        self.assertEqual(first['step'], second['step'])
        for key in ['model_state', 'optimizer_state', 'rng_state', 'cuda_rng_state', 'best_validation_nll',
                    'epoch', 'entry_cursor', 'batch_cursor']:
            assert_tree_equal(self, first[key], second[key], key)
        self.assertEqual(self.load(full, 'best.pt')['validation_metrics'],
                         self.load(interrupted, 'best.pt')['validation_metrics'])
        loaded = load_model(interrupted / 'best.pt', device='cpu')
        self.assertEqual(loaded.checkpoint_metadata['model_version'], MODEL_VERSION)
        self.assertTrue((interrupted / 'TRAINING_COMPLETE.json').exists())

    def test_max_steps_can_be_extended_without_resetting_optimizer(self):
        out = self.root / 'extend'
        self.run_training(out, max_steps=2)
        first = self.load(out)
        self.assertEqual(first['step'], 2)
        self.run_training(out, resume=True, max_steps=4)
        second = self.load(out)
        self.assertEqual(second['step'], 4)
        self.assertNotEqual(first['model_state']['pair_joint.3.weight'].tolist(),
                            second['model_state']['pair_joint.3.weight'].tolist())
        self.assertGreater(float(next(iter(second['optimizer_state']['state'].values()))['step']),
                           float(next(iter(first['optimizer_state']['state'].values()))['step']))

    def test_already_reached_step_limit_makes_no_extra_update(self):
        out = self.root / 'already_done'
        self.run_training(out, max_steps=2)
        before = self.load(out)
        self.run_training(out, resume=True, max_steps=2)
        after = self.load(out)
        self.assertEqual(after['step'], 2)
        assert_tree_equal(self, before['model_state'], after['model_state'])
        assert_tree_equal(self, before['optimizer_state'], after['optimizer_state'])

    def test_epochs_can_be_extended_with_same_semantic_identity(self):
        out = self.root / 'epochs'
        self.run_training(out, epochs=1)
        first = self.load(out)
        self.run_training(out, resume=True, epochs=2)
        second = self.load(out)
        self.assertEqual(first['identity_sha256'], second['identity_sha256'])
        self.assertEqual(second['epoch'], 2)
        self.assertEqual(second['step'], 2 * first['step'])

    def test_role_and_label_guards(self):
        with self.assertRaisesRegex(ValueError, 'role'):
            train.dataset(self.root / 'test.json', 'train')
        with self.assertRaisesRegex(ValueError, 'role'):
            train.dataset(self.root / 'train.json', 'validation')
        entry = train.dataset(self.root / 'train.json', 'train')['entries'][0]
        np.savez_compressed(entry['labels'], labels=np.full(8, 77))
        with self.assertRaisesRegex(ValueError, 'valid candidate genotype'):
            train.load_entry(entry)

    def test_changed_input_rejected_on_resume(self):
        out = self.root / 'identity'
        self.run_training(out, stop_after_checkpoint=1, success=False)
        with np.load(self.root / 'features0.npz', allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        values['depth'][0] += 1
        np.savez_compressed(self.root / 'features0.npz', **values)
        failed = self.run_training(out, resume=True, success=False)
        self.assertIn('identity changed', failed.stderr)

    def test_checkpoint_uses_safe_weights_only_types(self):
        out = self.root / 'weights_only'
        self.run_training(out, max_steps=1)
        for name in ['last.pt', 'best.pt']:
            state = self.load(out, name)
            self.assertIsInstance(state['config'], dict)
            self.assertIsInstance(state['model_state'], dict)
            self.assertIsInstance(state['rng_state'], torch.Tensor)
            self.assertEqual(state['model_version'], MODEL_VERSION)
        load_model(out / 'best.pt', device='cpu')
        self.assertEqual(self.load(out)['best_validation_nll'],
                         self.load(out, 'best.pt')['best_validation_nll'])


if __name__ == '__main__':
    unittest.main()
