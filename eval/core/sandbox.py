import os
import sys
import re
import tempfile
import types
import docker
import pandas as pd
import numpy as np

def format_result(total=0, status="FAIL", reason="", pillars=None, metrics=None, details=None):
    return {
        "total": total,
        "max": 100,
        "status": status,
        "reason": reason,
        "pillars": pillars or {},
        "metrics": metrics or {},
        "details": details or {}
    }

def _extract_clean_code(generated_code: str) -> str:
    code_match = re.search(r"```python\n(.*?)\n```", generated_code, re.DOTALL)
    return code_match.group(1) if code_match else generated_code

def run_in_docker_sandbox(code: str, data_map: dict, result_keys=['result_df', 'merged_df', 'output_df']) -> pd.DataFrame:
    try:
        client = docker.from_env()
    except docker.errors.DockerException:
        print("error: Docker is not available. Please ensure Docker is installed and running.")
        return None

    clean_code = _extract_clean_code(code)

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_loading_lines = []
        header_fix_lines = []
        input_names = list(data_map.keys())

        for var_name, df in data_map.items():
            var_dir_host = os.path.join(tmp_dir, var_name)
            os.makedirs(var_dir_host, exist_ok=True)
            df.to_csv(os.path.join(var_dir_host, "data.csv"), index=False)
            
            var_dir_container = f"/app/{var_name}"

            data_loading_lines.append(f"{var_name} = '{var_dir_container}'")
            data_loading_lines.append(f"os.environ['{var_name}'] = '{var_dir_container}'")
            
            header_fix_lines.append(f"{var_name} = os.path.join({var_name}, 'data.csv') if os.path.isdir(str({var_name})) else {var_name}")

        data_loading_str = "\n".join(data_loading_lines)
        header_fix_str = "\n".join(header_fix_lines)

 
        runner_code = f"""import pandas as pd
import numpy as np
import os
import sys

# variables and env vairbles
{data_loading_str}

# auto-fix header for path issue
{header_fix_str}

try:
    g = {{'pd': pd, 'np': np, 'os': os, 'sys': sys}}
    g.update({{k: v for k, v in locals().items() if not k.startswith('_')}})
    
    exec({repr(clean_code)}, g)
    
    # extract result
    target_df = None
    for key in {result_keys}:
        if key in g and isinstance(g[key], pd.DataFrame):
            target_df = g[key]
            break
            
    if target_df is None:
        input_names = {input_names}
        for k, v in g.items():
            if isinstance(v, pd.DataFrame) and k not in input_names:
                target_df = v
                break
                
    # output
    if target_df is not None:
        target_df.to_csv('/app/output.csv', index=False)

except Exception as e:
    import traceback
    with open('/app/error.log', 'w') as f:
        f.write(str(e) + "\\n\\n--- error details ---\\n" + traceback.format_exc())
"""
        with open(os.path.join(tmp_dir, "runner.py"), "w", encoding="utf-8") as f:
            f.write(runner_code)

        try:
            client.containers.run(
                image="eval-base:latest",
                command="python /app/runner.py",
                volumes={tmp_dir: {'bind': '/app', 'mode': 'rw'}},
                remove=True,
                network_disabled=True,
                mem_limit="512m"
            )
        except Exception as e:
            print(f"Docker exception: {e}")
            return None

        output_path = os.path.join(tmp_dir, "output.csv")
        error_path = os.path.join(tmp_dir, "error.log")
        
        if os.path.exists(output_path):
            return pd.read_csv(output_path)
        elif os.path.exists(error_path):
            with open(error_path, 'r') as f:
                print(f"LLM code error:\\n{f.read()}")
            return None
            
    return None
