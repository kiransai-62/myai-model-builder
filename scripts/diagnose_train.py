import tempfile, shutil, os, json
from pathlib import Path
from typer.testing import CliRunner
from myai.cli.main import app
from myai.core.config import ProjectConfig

runner = CliRunner()
temp_dir = Path(tempfile.mkdtemp())
try:
    (temp_dir / 'myai.yaml').write_text('project:\n  name: test\ngoal:\n  task: chat\n')
    (temp_dir / 'data').mkdir(parents=True, exist_ok=True)
    (temp_dir / 'data' / 'train.jsonl').write_text('{"prompt": "hi", "response": "hello"}\n')
    old = os.getcwd()
    os.chdir(temp_dir)
    res = runner.invoke(app, ['train', '--yes'])
    print('EXIT:', res.exit_code)
    print('STDOUT:\n', res.stdout)
    if res.exception:
        import traceback
        traceback.print_exception(type(res.exception), res.exception, res.exception.__traceback__)
finally:
    os.chdir(old)
    shutil.rmtree(temp_dir)
