import asyncio
import ast
import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.tcs_async_rl import gsm8k_driver
from experiments.tcs_async_rl import gsm8k_watch
from experiments.tcs_async_rl.gsm8k_control import (
    CONFIG, configure_bridge_cache, forwarded_runtime_environment, render_job, resolved_args,
    validate_derived_arguments,
)
from experiments.tcs_async_rl.rl_smoke import native_argv
from experiments.tcs_async_rl.gsm8k_wandb import acknowledge_evaluations, publish_evaluations
from experiments.tcs_async_rl.native_metrics import read_native_metrics
from experiments.tcs_async_rl.gsm8k_results import analyze, paired_accuracy


class GsmControlTests(unittest.TestCase):
    def test_watcher_verifies_the_same_snapshot_before_archiving(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'submission.json').write_text(json.dumps({'job_id':'1','remote_run':'/work/run'}))
            order = []
            def publish(*args):
                order.append('publish' if order.count('publish')==0 else 'verify')
                return {'verified_points':1 if 'verify' in order else 0}
            def archive(*args):
                order.append('archive')
                return 'https://example.org/immutable'
            with patch.object(gsm8k_watch,'remote_state',return_value={'state':'RUNNING'}), \
                 patch.object(gsm8k_watch,'sync_outputs',side_effect=lambda *a,**k: order.append('sync')), \
                 patch.object(gsm8k_watch,'read_native_metrics',return_value=[{'optimizer_step':0}]), \
                 patch.object(gsm8k_watch,'sync_native_wandb',side_effect=publish), \
                 patch.object(gsm8k_watch,'trace_paths',return_value=[types.SimpleNamespace(
                     stat=lambda:types.SimpleNamespace(st_size=64*2**20))]), \
                 patch.object(gsm8k_watch,'archive_chunks',side_effect=archive), \
                 patch.object(gsm8k_watch,'read_index',return_value={}):
                self.assertFalse(gsm8k_watch.cycle(root))
            self.assertEqual(order,['sync','publish','verify','archive'])
            self.assertEqual(json.loads((root/'monitor-heartbeat.json').read_text())['wandb']['verified_points'],1)

    def test_watcher_does_not_finish_unknown_jobs_and_stops_copying_at_low_space(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'submission.json').write_text(json.dumps({'job_id':'1','remote_run':'/work/run'}))
            with patch.object(gsm8k_watch,'remote_state',return_value={'state':'UNKNOWN'}), \
                 patch.object(gsm8k_watch,'sync_outputs') as sync, \
                 patch.object(gsm8k_watch,'read_native_metrics',return_value=[]), \
                 patch.object(gsm8k_watch,'trace_paths',return_value=[]), \
                 patch.object(gsm8k_watch,'read_index',return_value={}):
                self.assertFalse(gsm8k_watch.cycle(root))
                self.assertFalse((root/'monitor-complete.json').exists())
                sync.reset_mock()
                with patch.object(gsm8k_watch.shutil,'disk_usage',return_value=types.SimpleNamespace(free=2**30)):
                    with self.assertRaisesRegex(RuntimeError,'reserve'):
                        gsm8k_watch.cycle(root)
                sync.assert_not_called()

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

    def test_unscoped_baseline_and_startup_events_are_not_optimizer_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'events').mkdir()
            (root/'approved-config.json').write_text(json.dumps({'benchmark':'gsm8k'}))
            train = {'type':'metric','rollout_id':0,'metrics':{'train/pg_loss':0.1}}
            (root/'events/actor_cell00000_rank00000.jsonl').write_text(json.dumps(train)+'\n')
            events = [
                {'type':'metric','rollout_id':None,'metrics':{'eval/gsm8k':0.67,'eval/step':0}},
                {'type':'metric','rollout_id':None,'metrics':{'perf/init_time':3.0}},
                {'type':'metric','rollout_id':0,'metrics':{'rollout/episode_raw_reward':0.5}},
            ]
            path = root/'events/rollout_executor.jsonl'
            path.write_text('\n'.join(map(json.dumps,events))+'\n')
            self.assertEqual(read_native_metrics(root),[
                {'optimizer_step':0,'train/pg_loss':0.1,'rollout/episode_raw_reward':0.5}])
            with path.open('a') as stream:
                stream.write(json.dumps({'type':'metric','rollout_id':None,
                                         'metrics':{'rollout/episode_raw_reward':0.5}})+'\n')
            with self.assertRaisesRegex(ValueError,'no optimizer rollout ID'):
                read_native_metrics(root)

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
        self.assertNotIn('use_sampling_support_replay', args['cispo'])
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

    def test_control_cli_matches_native_options_and_derives_sampler_support(self):
        config = json.loads(CONFIG.read_text())
        fixture = json.loads((CONFIG.parent/'evidence/gsm8k_native_cli_options_20261128.json').read_text())
        source = CONFIG.parents[2]/'miles/utils/arguments.py'
        tree = ast.parse(source.read_text())
        literal = {a.value for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                   for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--')}
        defaults = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and argument.value in ('--rollout-top-p', '--rollout-top-k'):
                    defaults[argument.value[2:].replace('-', '_')] = ast.literal_eval(
                        next(keyword.value for keyword in node.keywords if keyword.arg == 'default'))
        registered = literal | set(fixture['frozen_external_option_names'])
        names = {'use_sampling_support_replay', 'rollout_sampling_logprobs_mode'}
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr in names
                       and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == 'args']
        self.assertEqual(len(assignments), 2)
        native_derivation = compile(ast.Module(body=assignments, type_ignores=[]), str(source), 'exec')
        for arm in ('reference', 'cispo'):
            args = resolved_args(config, arm=arm, work_dir=Path('/work'), run_dir=Path('/run'), run_id='check')
            flags = {value for value in native_argv(args) if value.startswith('--')}
            self.assertEqual(flags - registered, set())
            parsed = types.SimpleNamespace(**{**defaults, **args})
            exec(native_derivation, {'args': parsed})
            validate_derived_arguments(config, arm, parsed)
            self.assertEqual(parsed.use_sampling_support_replay, arm == 'cispo')
            self.assertEqual(parsed.rollout_sampling_logprobs_mode, 'support' if arm == 'cispo' else 'selected')
        parsed.use_sampling_support_replay = False
        with self.assertRaisesRegex(ValueError, 'approved setting'):
            validate_derived_arguments(config, 'cispo', parsed)

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
