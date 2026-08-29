import typer
from .init_cmd import init
from .system_cmd import check, benchmark
from .data_cmd import add, list_datasets, info, validate, prepare, clean, tokenize
from .model_cmd import list_models, add as add_model, use as use_model, list_trained, use_trained
from .recommend_cmd import recommend
from .train_cmd import train
from .index_cmd import build as build_index, add_document, list_indexes
from .export_cmd import export
from .ask_cmd import ask
from .serve_cmd import serve
from .evaluate_cmd import evaluate
from .optimize_cmd import optimize
from .auto_cmd import auto
from .status_cmd import status
from .ship_cmd import ship
from .merge_cmd import merge
from .reward_cmd import reward_app
from .runs_cmd import list_runs, info as run_info, best as runs_best

app = typer.Typer(help="MYAI: Local-first AI model builder", no_args_is_help=True)

app.command("init")(init)
app.command("status")(status)
app.command("auto")(auto)
app.command("recommend")(recommend)
app.command("train")(train)
app.command("optimize")(optimize)
app.command("export")(export)
app.command("ship")(ship)
app.command("merge")(merge)
app.command("ask")(ask)
app.command("serve")(serve)
app.command("evaluate")(evaluate)
app.command("leaderboard")(runs_best)

data_app = typer.Typer(help="Data operations")
data_app.command("add")(add)
data_app.command("list")(list_datasets)
data_app.command("info")(info)
data_app.command("validate")(validate)
data_app.command("prepare")(prepare)
data_app.command("clean")(clean)
data_app.command("tokenize")(tokenize)
app.add_typer(data_app, name="data")

model_app = typer.Typer(help="Model operations")
model_app.command("list")(list_models)
model_app.command("add")(add_model)
model_app.command("use")(use_model)
model_app.command("trained")(list_trained)
model_app.command("use-trained")(use_trained)
app.add_typer(model_app, name="model")

system_app = typer.Typer(help="System operations")
system_app.command("check")(check)
system_app.command("benchmark")(benchmark)
app.add_typer(system_app, name="system")

index_app = typer.Typer(help="Knowledge index operations")
index_app.command("build")(build_index)
index_app.command("add")(add_document)
index_app.command("list")(list_indexes)
app.add_typer(index_app, name="index")

runs_app = typer.Typer(help="Training run provenance")
runs_app.command("list")(list_runs)
runs_app.command("info")(run_info)
runs_app.command("best")(runs_best)
app.add_typer(runs_app, name="runs")

app.add_typer(reward_app, name="reward")

if __name__ == "__main__":
    app()