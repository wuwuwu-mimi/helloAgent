import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class HelloAgentLLM:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None, timeout: int = 30):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout

        if not all([self.model, self.api_key, self.base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        print(f"正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            collected_content = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


class MyLLM(HelloAgentLLM):
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 provider: Optional[str] = None, **kwargs):
        self.provider = provider or self._auto_detect_provider(base_url, api_key)
        resolved_api_key, resolved_base_url = self._resolve_credentials(base_url, api_key)

        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens')
        self.timeout = kwargs.get('timeout', 60)

        if resolved_api_key and resolved_base_url and self.model:
            self.api_key = resolved_api_key
            self.base_url = resolved_base_url
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                timeout=self.timeout
            )
        else:
            super().__init__(
                model=model,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                timeout=self.timeout
            )

    def _auto_detect_provider(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> str:
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("ZHIPU_API_KEY"):
            return "zhipu"
        if os.getenv("MOONSHOT_API_KEY"):
            return "moonshot"
        if os.getenv("MINIMAX_API_KEY"):
            return "minimax"
        if os.getenv("DOUBAN_API_KEY"):
            return "douban"

        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            u = actual_base_url.lower()
            if "api.openai.com" in u:
                return "openai"
            if "api.deepseek.com" in u:
                return "deepseek"
            if "dashscope.aliyuncs.com" in u:
                return "qwen"
            if "open.bigmodel.cn" in u:
                return "zhipu"
            if "api.moonshot.cn" in u:
                return "moonshot"
            if "ark.cn-beijing.volces.com" in u:
                return "douban"
            if "localhost" in u or "127.0.0.1" in u:
                if ":11434" in u:
                    return "ollama"
                if ":8000" in u:
                    return "vllm"
                return "local"
            if "proxy" in u or "ai" in u:
                return "openai"

        return "auto"

    def _resolve_credentials(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> tuple[str, str]:

        """
        根据 provider 得到 api_key and Base_url
        :param base_url:
        :param api_key:
        :return: api_key and Base_url
        """
        provider = self.provider.lower()
        resolved_api_key = ""
        resolved_base_url = ""

        if provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"

        elif provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1"

        elif provider == "qwen":
            resolved_api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv(
                "LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        elif provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"

        elif provider == "moonshot":
            resolved_api_key = api_key or os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"

        elif provider == "douban":
            resolved_api_key = api_key or os.getenv("DOUBAN_API_KEY") or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"

        elif provider in ["local", "ollama", "vllm"]:
            resolved_api_key = api_key or "fake-key"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://127.0.0.1:11434/v1"

        else:
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or ""
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"

        return resolved_api_key.strip() if resolved_api_key else "", resolved_base_url.strip() if resolved_base_url else ""


if __name__ == "__main__":
    llm = MyLLM()
    llm.think([{"role": "user", "content": "你好"}])