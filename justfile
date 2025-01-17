alias ta := test-all
alias tl := test-lowest-versions
alias th := test-highest-versions
alias tt := test-nogil
alias t := test
alias w := watch
alias p := tag-and-publish
alias d := docs
alias s := serve
alias r := recover-uvlock
alias f := fmt
alias c := clean

@default:
    just -u -l

watch *flags:
    uv run --frozen -- watchmedo shell-command \
      --patterns='*.py' \
      --recursive \
      --command='just test {{ flags }}' \
      --verbose \
      src/ tests/

# Disable warnings by adding `--disable-warnings`.
test *flags:
    uv run --frozen -- \
    pytest -s tests/ {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}

test-all *flags:
    just test-lowest-versions {{ flags }}
    just test-highest-versions {{ flags }}
    just test-nogil {{ flags }}

test-lowest-versions *flags:
    uv run --resolution=lowest-direct --python=3.10 --isolated -- \
    pytest -s {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}
    just recover-uvlock

test-highest-versions *flags:
    uv run --resolution=highest --python=3.13 --isolated -- \
    pytest -s {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}
    just recover-uvlock

test-nogil *flags:
    uv run --resolution=highest --python=3.13t --isolated -- \
    pytest -s {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}
    just recover-uvlock

sync:
    uv sync --frozen

python:
    uv run --frozen -- ipython --no-autoindent

docs:
    rm -rf docs/*
    uv run --frozen pdoc --math ./src/arianna -o ./docs --docformat numpy

serve:
    # uv run --frozen python -m http.server -d ./docs 8000
    uv run --frozen pdoc --docformat numpy --math -p 8000 ./src/arianna

fmt-self:
    just --fmt --unstable

recover-uvlock:
    git checkout uv.lock
    uv sync --frozen

# uv run --resolution=lowest-direct --python=3.10 --isolated -- pytest -s

fmt-py:
    ruff format

fmt: fmt-self fmt-py

lint:
    ruff check --fix

clean:
    rm -rf src/*.egg-info src/arianna/__pycache__ src/arianna/*/__pycache__
    rm -rf dist

build:
    uv build

publish: clean build
    uv publish

tag-and-publish bump:
    uv run bumpy {{ bump }} -p
    just publish
