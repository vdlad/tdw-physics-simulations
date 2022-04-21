import json
from pathlib import Path
from tdw.controller import Controller
from tdw.output_data import OutputData

c = Controller(launch_build=True)
p = Path.home().joinpath("Downloads/tdw_commands (1).json")
txt = p.read_text()
for i, line in enumerate(txt.split("\n")):
    try:
        if not line.startswith("["):
            continue
        commands = json.loads(line.split(" trial ")[0])
        if not isinstance(commands, list):
            continue
        resp = c.communicate(commands)
        r_ids = [OutputData.get_data_type_id(r) for r in resp[:-1]]
        assert "imag" in r_ids, (i, line, r_ids)
    except json.JSONDecodeError:
        continue
