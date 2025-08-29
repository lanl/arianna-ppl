set windows-shell := ['powershell.exe', '-NoProfile', '-Command']

alias ta := test-all
alias tl := test-lowest-versions
alias th := test-highest-versions
alias tt := test-nogil
alias t := test
alias w := watch
alias d := docs
alias s := serve
alias c := clean
alias b := bump
alias l := lint

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
    uv run --frozen \
        pytest -s tests/ {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}

test-all *flags:
    just test-lowest-versions {{ flags }}
    just test-highest-versions {{ flags }}
    just test-nogil {{ flags }}

test-lowest-versions *flags:
    uv run \
        --resolution=lowest-direct --python=3.10 --isolated \
        pytest -s {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}

test-highest-versions *flags:
    uv run --resolution=highest --python=3.13 --isolated \
        pytest -s {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}

test-nogil *flags:
    uv run --resolution=highest --python=3.13t --isolated \
        pytest -s {{ if flags == "-dw" { "--disable-warnings" } else { "" } }}

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

lint:
    uv run pre-commit run -a

clean:
    rm -rf src/*.egg-info src/arianna/__pycache__ src/arianna/*/__pycache__
    rm -rf dist

# Update git tag and push tag. GitHub Actions will then publish to PyPI.
bump kind:
    uv run bump {{ kind }} -p
