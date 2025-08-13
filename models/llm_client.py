import subprocess
from typing import Dict, Any, List

class OllamaClient:
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Minimal wrapper using `ollama run` to keep it hackathon-simple
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        cmd = ["ollama", "run", self.model]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(prompt)
        if p.returncode != 0:
            raise RuntimeError(err)
        return out
