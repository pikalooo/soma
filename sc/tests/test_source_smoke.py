from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "submodular_gpt5"


def test_all_python_sources_parse():
    for path in SRC.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_deepseek_or_hardcoded_key_config():
    source = "\n".join(
        p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
    )
    assert "DEEPSEEK_BASE_URL" not in source
    assert "DEEPSEEK_API_KEY" not in source
    assert "api.deepseek.com" not in source


def test_gpt5_and_responses_api_are_default():
    llm = (SRC / "llm.py").read_text(encoding="utf-8")
    cfg = (SRC / "config.py").read_text(encoding="utf-8")
    assert "responses.create" in llm
    assert '"gpt-5"' in cfg


def test_removed_compatibility_parameters_are_not_public_constructor_args():
    tree = ast.parse((SRC / "memory" / "static.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "StaticSubmodularMemoryCompressor")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "__init__")
    args = {x.arg for x in init.args.args + init.args.kwonlyargs}
    removed = {
        "lambda_cost",
        "diversity_method",
        "min_gain",
        "lazy_greedy",
        "use_ratio",
        "max_features",
        "ngram_range",
        "ftgp_alpha",
        "granular_beta",
    }
    assert args.isdisjoint(removed)
