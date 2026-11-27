import asyncio
import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.tcs_async_rl import gsm8k_driver
from experiments.tcs_async_rl.gsm8k_control import (
    CONFIG, configure_bridge_cache, forwarded_runtime_environment, render_job, resolved_args,
)
from experiments.tcs_async_rl.gsm8k_wandb import acknowledge_evaluations, publish_evaluations
from experiments.tcs_async_rl.native_metrics import read_native_metrics
from experiments.tcs_async_rl.gsm8k_results import analyze, paired_accuracy


class GsmControlTests(unittest.TestCase):
    def test_bridge_cache_uses_work_storage_and_reaches_workers_without_home_or_keys(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(os.environ, {}, clear=True):
            directory = configure_bridge_cache(Path(name))
            self.assertTrue(directory.is_dir())
            environment = {**os.environ, 'HOME':'/read-only/home', 'HF_HOME':'/work/hf',
                           'OPENAI_API_KEY':'synthetic-placeholder', 'TRITON_CACHE_DIR':'/work/triton'}
            forwarded = forwarded_runtime_environment(environment)
            self.assertEqual(forwarded['MEGATRON_CONFIG_LOCK_DIR'], str(directory))
            self.assertEqual(forwarded['HF_HOME'], '/work/hf')
            self.assertEqual(forwarded['TRITON_CACHE_DIR'], '/work/triton')
            self.assertNotIn('HOME', forwarded)
            self.assertNotIn('OPENAI_API_KEY', forwarded)

    def test_terminal_analysis_allows_untrained_prefetch_but_requires_final_weight_publication(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'events').mkdir()
            (root/'evaluations').mkdir()
            (root/'approved-config.json').write_text(json.dumps({'benchmark':'gsm8k'}))
            (root/'control-complete.json').write_text(json.dumps({'uninformative':False,'optimizer_updates':1}))
            for filename, updates, scores in [('baseline',0,{'1':0,'2':1}),
                                              ('updates-000001',1,{'1':1,'2':1})]:
                (root/'evaluations'/f'{filename}.json').write_text(json.dumps({
                    'optimizer_updates':updates,'scores':scores}))
            train = {'type':'metric','rollout_id':0,'metrics':{
                'train/pg_loss':0.1,'perf/update_weights_time':0.1,
                'train/update_direction/surrogate_gain':0.01}}
            (root/'events/actor_cell00000_rank00000.jsonl').write_text(json.dumps(train)+'\n')
            rollout = [{'type':'metric','rollout_id':i,'metrics':{'rollout/episode_raw_reward':0.5}}
                       for i in (0,1)]
            (root/'events/rollout_executor.jsonl').write_text('\n'.join(map(json.dumps,rollout))+'\n')
            publication = root/'events/main.jsonl'
            publication.write_text(json.dumps({'type':'inference_engine_weight_checksum','rollout_id':0})+'\n')
            result = analyze(root)
            self.assertEqual(result['optimizer_updates'],1)
            self.assertTrue(result['gate_passed'])
            publication.write_text('')
            with self.assertRaisesRegex(ValueError,'optimizer history differ'):
                analyze(root)

    def test_paired_analysis_distinguishes_no_change_and_unanimous_improvement(self):
        before = {str(i):0 for i in range(8)}
        unchanged = paired_accuracy(before, before, bootstrap_samples=100)
        self.assertEqual(unchanged['change'], 0)
        self.assertEqual(unchanged['mcnemar_two_sided_exact_p'], 1)
        self.assertEqual(unchanged['paired_problem_bootstrap_95_interval'], [0, 0])
        improved = paired_accuracy(before, {key:1 for key in before}, bootstrap_samples=100)
        self.assertEqual(improved['change'], 1)
        self.assertEqual(improved['mcnemar_two_sided_exact_p'], 2/256)
        self.assertEqual(improved['bonferroni_two_comparisons_p'], 4/256)
        with self.assertRaises(ValueError):
            paired_accuracy(before, {'other':1})

    def test_evaluation_history_has_its_own_axis_and_acknowledged_points_are_not_resent(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'evaluations').mkdir()
            for updates, score in [(0, 0.4), (20, 0.5)]:
                (root/'evaluations'/f'{updates:06d}.json').write_text(json.dumps({
                    'optimizer_updates':updates, 'accuracy':score, 'valid_box_fraction':1,
                    'capped_fraction':0, 'mean_response_tokens':100}))
            run = Mock()
            delivered = publish_evaluations(run, root)
            self.assertEqual(run.log.call_count, 2)
            self.assertNotIn('audit/optimizer_step', run.log.call_args.args[0])
            self.assertEqual(run.log.call_args.args[0]['control/eval_optimizer_updates'], 20)
            acknowledge_evaluations(root, delivered)
            run.reset_mock()
            publish_evaluations(run, root)
            run.log.assert_not_called()

    def test_late_evaluation_metrics_cannot_mutate_a_published_training_row(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'events').mkdir()
            (root/'approved-config.json').write_text(json.dumps({'benchmark':'gsm8k'}))
            train = {'type':'metric','rollout_id':0,'metrics':{'train/pg_loss':0.1}}
            rollout = {'type':'metric','rollout_id':0,'metrics':{'rollout/episode_raw_reward':0.4}}
            (root/'events/actor_cell00000_rank00000.jsonl').write_text(json.dumps(train)+'\n')
            p = root/'events/rollout_executor.jsonl'
            p.write_text(json.dumps(rollout)+'\n')
            before = read_native_metrics(root)
            with p.open('a') as stream:
                stream.write(json.dumps({'type':'metric','rollout_id':0,'metrics':{'eval/gsm8k':0.5}})+'\n')
            self.assertEqual(before, read_native_metrics(root))

    def test_both_arms_start_from_identical_pinned_weights_and_keep_approved_batches(self):
        config = json.loads(CONFIG.read_text())
        args = {arm: resolved_args(config, arm=arm, work_dir=Path('/work'),
                                  run_dir=Path('/work') / arm, run_id=arm)
                for arm in ('reference', 'cispo')}
        self.assertEqual(args['reference']['load'], args['cispo']['load'])
        for arm, batch in [('reference', 256), ('cispo', 128)]:
            a = args[arm]
            self.assertEqual(a['global_batch_size'], batch)
            self.assertEqual(a['rollout_batch_size'] * a['n_samples_per_prompt'], batch)
            self.assertEqual(a['rollout_max_response_len'], 1024)
            self.assertEqual(a['num_rollout'], 250)
            self.assertTrue(a['finetune'])
            self.assertEqual(a['ref_load'], a['load'])
            self.assertEqual(a['rm_type'], 'math')
            self.assertTrue(a['rollout_shuffle'])
            self.assertNotIn('custom_rm_path', a)
            self.assertEqual(a['save'], '/work/' + arm + '/checkpoints')
        self.assertFalse(args['reference']['fully_async'])
        self.assertTrue(args['cispo']['fully_async'])
        self.assertTrue(args['cispo']['use_sampling_support_replay'])
        unapproved = copy.deepcopy(config)
        unapproved['status'] = 'proposed_requires_plan_approval'
        with self.assertRaises(ValueError):
            resolved_args(unapproved, arm='reference', work_dir=Path('/w'), run_dir=Path('/w/r'), run_id='x')

    def test_retry_walltime_rounds_down_and_never_exceeds_remaining_arm_budget(self):
        config = json.loads(CONFIG.read_text())
        kwargs = dict(repo=Path('/repo'), work_dir=Path('/work'), run_dir=Path('/work/run'),
                      config=config, config_path=CONFIG, arm='reference', expected_commit='abc')
        self.assertIn('#SBATCH --time=01:59:00', render_job(**kwargs, seconds=7199))
        for seconds in (0, 59, 7201):
            with self.assertRaises(ValueError):
                render_job(**kwargs, seconds=seconds)

    def driver_fixture(self, root, baseline, *, fail=False):
        calls = []
        controller = types.SimpleNamespace()
        async def prepare_eval():
            calls.append('prepare_final')
        controller.prepare_eval = prepare_eval
        async def update(*args, **kwargs):
            calls.append(('update', kwargs.get('rollout_id')))
        class Dispatcher:
            def __init__(self, *args):
                pass
            async def dispatch(self, rollout_id, **kwargs):
                calls.append(('eval', rollout_id))
        native = types.SimpleNamespace(update_weights=update, EvalDispatcher=Dispatcher)
        async def train(args, *, disposer):
            await native.update_weights(args, None, None, controller)
            dispatcher = native.EvalDispatcher(args, None, None)
            await dispatcher.dispatch(0)
            if fail:
                raise RuntimeError('real failure')
            for step in range(3):
                await native.update_weights(args, None, None, controller, rollout_id=step)
        native.train = train
        args = types.SimpleNamespace(save_debug_rollout_data=str(root/'rollout_data/rollout-{rollout_id}.pt'))
        def export(*args, **kwargs):
            return {'accuracy': baseline if kwargs['optimizer_updates'] == 0 else 0.6}
        return args, native, calls, export

    def test_deadline_exit_gets_a_final_eval_before_disposal_without_replacing_training(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, native, calls, export = self.driver_fixture(root, 0.4)
            original = native.train
            with patch.dict('sys.modules', {'train_async': native}), patch.object(gsm8k_driver, 'export_evaluation', export):
                asyncio.run(gsm8k_driver.train(args, disposer=None))
            self.assertIs(native.train, original)
            self.assertEqual(calls[-2:], ['prepare_final', ('eval', 2)])
            self.assertEqual(json.loads((root/'control-complete.json').read_text())['optimizer_updates'], 3)

    def test_ceiling_gate_prevents_any_optimizer_update_and_real_failures_stay_failures(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, native, calls, export = self.driver_fixture(root, 0.96)
            with patch.dict('sys.modules', {'train_async': native}), patch.object(gsm8k_driver, 'export_evaluation', export):
                asyncio.run(gsm8k_driver.train(args, disposer=None))
            self.assertEqual([x for x in calls if isinstance(x, tuple) and x[0] == 'update'], [('update', None)])
            self.assertTrue(json.loads((root/'control-complete.json').read_text())['uninformative'])
        with tempfile.TemporaryDirectory() as name:
            args, native, calls, export = self.driver_fixture(Path(name), 0.4, fail=True)
            with patch.dict('sys.modules', {'train_async': native}), patch.object(gsm8k_driver, 'export_evaluation', export):
                with self.assertRaisesRegex(RuntimeError, 'real failure'):
                    asyncio.run(gsm8k_driver.train(args, disposer=None))


if __name__ == '__main__':
    unittest.main()
