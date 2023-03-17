import json
import subprocess

import src.utils.general as utils
from . import BaseLauncher

log = utils.get_logger(__name__)


class LocalLauncher(BaseLauncher):
    def __init__(self, work_dir):
        super().__init__()
        self.work_dir = work_dir

    def prepare_cmd(self, script, config):
        overrides = " ".join([f"{key}={value}" for key, value in config.items() if not key.startswith('--')])
        other = " ".join([f"{key} {value}" for key, value in config.items() if key.startswith('--')])
        if not other:
            return f"{script} {overrides}", None
        cmd = f"{script} --overrides {overrides} {other}"
        return cmd, config['--exp_dir']

    def run_trial(self, idx, script_1, config_1, script_2=None, config_2=None):
        # result_code = os.system(f"python tests/sleep.py --id {idx}")
        # result_code = subprocess.run(["python", 'tests/sleep.py', '--id', f'{idx}'])

        # args = " ".join([f"{key}={value}" for key, value in config_1.items()])
        # cmd = f"{script_1} {args}"
        cmd, eval_path = self.prepare_cmd(script_1, config_1)
        log.info(f"[Trial {idx}] Executing command: {cmd}")
        result_code = subprocess.run(cmd.split(" "), cwd=self.work_dir)

        if script_2 is not None:
            # note that subprocess.run waits until the command passes to it finishes, so we
            # can safely continue the evaluation subprocess.
            # args = " ".join([f"{key}={value}" for key, value in config_2.items()])
            # cmd = f"{script_2} {args}"
            cmd, eval_path = self.prepare_cmd(script_2, config_2)
            log.info(f"[Trial {idx}] Executing command: {cmd}")
            result_code = subprocess.run(cmd.split(" "), cwd=self.work_dir)

        # TODO: Verify that best_chkpoint file exists for the training
        # If there is evaluation, verify that results.json exists (if run was successful)

        if eval_path:
            with open(eval_path, 'r') as f:
                results = json.load(f)

        # TODO: Read the score from json file inside evaluation directory and return it
        # What to do if no eval?
        print(results)
        score = 0.
        # score = results['unbiased-small']['corpus']
        # score = results['BLEU-4__tok-13a_sm-_exp_sv_None']['corpus']
        return score