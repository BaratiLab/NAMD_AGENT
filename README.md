# NAMD Agent: HTMD/NAMD3 MCP workflow

This repository packages a Model Context Protocol (MCP) server for building,
running, and analysing molecular-dynamics workflows with HTMD/MoleculeKit and
NAMD3. The server supports protein-in-water, protein-ligand, protein-protein,
protein-nucleic-acid, and protein-membrane systems, plus trajectory analysis,
comparison, QA, and PyMOL/ffmpeg visualisation tools.

## Repository contents

| File | Purpose |
|---|---|
| `htmd-mcp-server.zip` | MCP server source, NAMD templates, and implementation modules |
| `requirements.txt` | Conda MatchSpec file for recreating the tested software environment |
| `.gitignore` | Excludes extracted runtime files |
| `LICENSE` | Repository license |

NAMD3 is not distributed in this repository. Obtain a suitable NAMD3 build
separately and comply with its license.

## Prerequisites

- Linux x86-64 is recommended; the supplied requirements were tested on this
  platform.
- Conda is required when creating the environment from `requirements.txt`.
- `unzip` is needed to extract the MCP server.
- NAMD3 is required to execute simulations; a GPU build is recommended for
  production work.
- Network access is required for optional RCSB PDB downloads and SwissParam
  ligand parameterisation.
- `psfgen` is required only by tools that generate lipid PDBs.

## Recommended setup: create the Conda environment

The root `requirements.txt` is a **Conda requirements file**, not a pip
requirements file. Create the tested environment with Conda and both required
channels:

```bash
git clone https://github.com/BaratiLab/NAMD_AGENT.git
cd NAMD_AGENT

conda create --name namd-agent \
  --override-channels \
  --channel acellera \
  --channel conda-forge \
  --file requirements.txt
conda activate namd-agent
```

Do not run `pip install -r requirements.txt`; channel-qualified Conda package
specifications are intentionally used because HTMD and MoleculeKit are not a
pip-only stack.

Check the core imports and executables:

```bash
python -c "import htmd, moleculekit, mcp; print('HTMD/MCP imports OK')"
command -v obabel
command -v pymol
command -v ffmpeg
```

The versions are pinned to the supplied environment for reproducibility. If a
pin later becomes unavailable, relax only that pin and record the resulting
`conda list --explicit` output with the run.

## Extract the MCP server

From the repository root:

```bash
mkdir -p runtime
unzip -q htmd-mcp-server.zip -d runtime

export HTMD_MCP_DIR="$PWD/runtime/htmd-mcp-server"
export HTMD_ENV_DIR="$CONDA_PREFIX"
```

The archive extracts to `runtime/htmd-mcp-server/` and contains:

```text
main.py                 stdio MCP server and tool definitions
functions.py            command-line function dispatcher
htmd_mcp/               builders, runners, analyses, QA, and visualisation
htmd_mcp/templates/     NAMD3 configuration templates
requirements.txt        original server dependency notes
README.md               server-specific tool notes
```

Run non-blocking compile and import checks:

```bash
"$HTMD_ENV_DIR/bin/python" -m py_compile \
  "$HTMD_MCP_DIR/main.py" \
  "$HTMD_MCP_DIR/functions.py"

PYTHONPATH="$HTMD_MCP_DIR" \
  "$HTMD_ENV_DIR/bin/python" -c \
  "import main; print('MCP server import OK')"
```

Do not start `main.py` directly as a smoke test: it opens a stdio MCP loop and
waits for a client.

## Install and identify NAMD3

Download NAMD3 from the official NAMD distribution and select a build compatible
with the host. GPU builds must also match the installed NVIDIA driver/CUDA
stack.

```bash
export NAMD3_BIN="/absolute/path/to/namd3"
test -x "$NAMD3_BIN"
"$NAMD3_BIN" --version 2>&1 | head
```

Some NAMD builds do not accept `--version`; invoking the executable without a
configuration file should still print its version banner. Validate the target
machine with a short minimisation or benchmark before starting a production run.

If a workflow invokes external `psfgen`, also set:

```bash
export PSFGEN_BIN="/absolute/path/to/psfgen"
```

## Configure the MCP server

Set executable and cache paths before launching the MCP client:

```bash
export HTMD_PYTHON="$HTMD_ENV_DIR/bin/python"
export NAMD3_BIN="/absolute/path/to/namd3"
export OBABEL_BIN="$HTMD_ENV_DIR/bin/obabel"
export PYMOL_BIN="$HTMD_ENV_DIR/bin/pymol"
export FFMPEG_BIN="$HTMD_ENV_DIR/bin/ffmpeg"
export HTMD_MCP_LOG="INFO"
export HTMD_MCP_SWISSPARAM_CACHE="$PWD/runtime/swissparam_cache"
mkdir -p "$HTMD_MCP_SWISSPARAM_CACHE"
```

The server recognises these variables:

| Variable | Purpose |
|---|---|
| `HTMD_PYTHON` | Python interpreter used by `main.py` for tool dispatch |
| `NAMD3_BIN` | NAMD3 executable |
| `PSFGEN_BIN` | External psfgen executable for lipid-PDB generation |
| `OBABEL_BIN` | Open Babel command-line executable |
| `PYMOL_BIN` | PyMOL executable used for rendering |
| `FFMPEG_BIN` | ffmpeg executable used for movies |
| `SWISSPARAM_BASE` | SwissParam endpoint; defaults to `https://www.swissparam.ch:8443` |
| `HTMD_MCP_SWISSPARAM_CACHE` | Local ligand-parameter cache |
| `HTMD_MCP_LOG` | Server log level; defaults to `INFO` |

`HTMD_PYTHON` and `NAMD3_BIN` are the essential overrides for most workflows.
The source contains historical machine-specific defaults, so explicit absolute
paths are strongly recommended.

## Register the stdio MCP server

Add an entry like this to the JSON configuration used by your MCP client. Replace
every placeholder with an absolute path; many clients do not expand shell
variables or `~` in their configuration.

```json
{
  "mcpServers": {
    "namd-agent": {
      "type": "stdio",
      "command": "/absolute/path/to/environment/bin/python",
      "args": [
        "/absolute/path/to/NAMD_AGENT/runtime/htmd-mcp-server/main.py"
      ],
      "env": {
        "HTMD_PYTHON": "/absolute/path/to/environment/bin/python",
        "NAMD3_BIN": "/absolute/path/to/namd3",
        "OBABEL_BIN": "/absolute/path/to/environment/bin/obabel",
        "PYMOL_BIN": "/absolute/path/to/environment/bin/pymol",
        "FFMPEG_BIN": "/absolute/path/to/environment/bin/ffmpeg",
        "HTMD_MCP_SWISSPARAM_CACHE": "/absolute/path/to/NAMD_AGENT/runtime/swissparam_cache",
        "HTMD_MCP_LOG": "INFO"
      }
    }
  }
}
```

Restart the MCP client after editing its configuration. The client launches the
server and communicates over standard input/output; diagnostic text is written
to standard error.

## Direct dispatcher smoke test

Every MCP operation can also be called through `functions.py`. This example
downloads ubiquitin from the RCSB PDB service and therefore requires network
access:

```bash
mkdir -p runtime/smoke_test
cd runtime/htmd-mcp-server

"$HTMD_ENV_DIR/bin/python" functions.py fetch_pdb \
  '{"pdb_id":"1UBQ","out_dir":"../smoke_test"}'
```

Use the compile/import checks above when testing offline.

## Operational notes

- Use a new run directory or output basename so archived results are not
  overwritten.
- Confirm every topology/parameter path resolves before launching NAMD3.
- A trajectory must be analysed with a PSF containing the same atom count and
  ordering. Wrapped/aligned stripped trajectories may need a matching subset
  topology.
- Some build tools can print an HTMD chain report before their JSON result. A
  client that requires JSON-only standard output may misdiagnose the call.
- For long runs, inspect NAMD logs for temperature, energy, cell dimensions,
  constraint errors, and the normal completion marker.

## Troubleshooting

- **Conda cannot solve the environment:** confirm the channel order and that both
  `acellera` and `conda-forge` are reachable. Do not substitute pip packages for
  HTMD or MoleculeKit.
- **The MCP client disconnects immediately:** run the compile/import checks,
  inspect client stderr, and confirm all configured paths are absolute.
- **NAMD3 is not found:** set `NAMD3_BIN` in the MCP client's `env` object.
- **CUDA fails:** use a NAMD3 build compatible with the host driver/CUDA stack,
  or validate with a CPU build.
- **SwissParam or PDB fetching fails:** check outbound network and TLS access.
- **Trajectory/topology mismatch:** use the matching PSF stored with the original
  trajectory.
