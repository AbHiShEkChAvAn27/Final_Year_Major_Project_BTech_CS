"""
Core agent implementation using LangGraph
"""

import os
import re
import json
import time
import asyncio
import subprocess
from datetime import datetime
from typing import Dict
from pathlib import Path
import paramiko

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .state import ProjectState
from .prompts import get_system_prompt
from ..knowledge.manager import EmbeddedSystemsTools
from ..tools import get_all_tools


class EmbeddedSystemsAgent:
    """Main agent class using LangGraph for embedded systems development"""
    
    def __init__(self, api_key: str = None, knowledge_base_path: str = "./knowledge_base"):
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_api_key:
            self.llm = ChatNVIDIA(
                model="meta/llama3-70b-instruct",
                api_key=nvidia_api_key,
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
            )
            print("✅ Using NVIDIA Llama 3 endpoint for LLM")
        else:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-2.5-flash",
                temperature=0.1,
                convert_system_message_to_human=True,
                timeout=120,
                max_retries=2,
            )
            print("✅ Using Gemini endpoint for LLM")

        self.tools_instance = EmbeddedSystemsTools(knowledge_base_path)
        self.tools = get_all_tools()
        self.tool_node = ToolNode(tools=self.tools)
        self.graph = self._create_graph()
        self._max_iterations = 3

    def _get_content_as_string(self, content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif hasattr(item, 'text'):
                    parts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
            return '\n'.join(parts)
        return str(content)

    def _create_graph(self) -> StateGraph:
        def call_agent(state: ProjectState):
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            if state["iteration_count"] > self._max_iterations:
                return state
            
            response = self.llm.bind_tools(self.tools).invoke(state["messages"])
            state["messages"].append(response)
            
            if hasattr(response, 'content') and response.content:
                code = self._extract_code_from_response(self._get_content_as_string(response.content))
                if code:
                    state["generated_code"] = code
            return state

        def should_use_tools(state: ProjectState):
            if state.get("iteration_count", 0) >= self._max_iterations:
                return "end"
            last_msg = state["messages"][-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                return "tools"
            return "end"

        def call_tools(state: ProjectState):
            result = self.tool_node.invoke({"messages": state["messages"]})
            if isinstance(result, dict) and "messages" in result:
                state["messages"] = result["messages"]
            return state

        def should_continue_after_tools(state: ProjectState):
            return "end" if state.get("iteration_count", 0) >= self._max_iterations else "agent"

        workflow = StateGraph(ProjectState)
        workflow.add_node("agent", call_agent)
        workflow.add_node("tools", call_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_use_tools, {"tools": "tools", "end": END})
        workflow.add_conditional_edges("tools", should_continue_after_tools, {"agent": "agent", "end": END})

        return workflow.compile()

    async def add_knowledge(self, file_path: str) -> bool:
        try:
            path = Path(file_path)
            if path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(path))
            elif path.suffix.lower() == '.txt':
                loader = TextLoader(str(path))
            else:
                return False

            documents = loader.load()
            texts = self.tools_instance.text_splitter.split_documents(documents)
            self.tools_instance.vectorstore.add_documents(texts)
            print(f"Added {len(texts)} chunks from {path.name}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    async def process_request(self, user_input: str, platform: str = "", max_retries: int = 3, mode: str = "default", ssh_config: dict = None, board_config: dict = None) -> Dict:
        """
        Handles both normal and board chat requests. Use mode="board" for board-specific chat.
        """
        # Support legacy ssh_config parameter
        if board_config is None and ssh_config is not None:
            board_config = {"type": "ssh", **ssh_config}
        if mode == "board":
            return await self.process_board_request(user_input, platform, max_retries, board_config=board_config)
        
        NETWORK_ERRORS = ["10054", "10053", "10060", "ConnectionResetError", "ConnectionError", "TimeoutError"]
        
        for attempt in range(max_retries):
            try:
                initial_state = ProjectState(
                    messages=[
                        SystemMessage(content=get_system_prompt(platform)),
                        HumanMessage(content=f"Platform: {platform}\nRequest: {user_input}")
                    ],
                    platform=platform,
                    requirements=user_input,
                    current_step="processing",
                    iteration_count=0
                )

                result = await asyncio.get_event_loop().run_in_executor(None, self.graph.invoke, initial_state)
                final_message = result["messages"][-1]
                response_content = self._get_content_as_string(getattr(final_message, 'content', str(final_message)))

                return {"success": True, "response": response_content, "platform": platform, "timestamp": datetime.now().isoformat()}

            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(60)
                        continue
                    return {"success": False, "error": "API rate limit exceeded. Please wait and try again."}
                
                if any(err in error_str or err in type(e).__name__ for err in NETWORK_ERRORS):
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 5)
                        continue
                    return {"success": False, "error": f"Network error. Check your connection.\n{error_str}"}
                
                return {"success": False, "error": f"Request failed: {error_str}"}
        
        return {"success": False, "error": "Request failed after retries."}

    async def process_board_request(self, user_input: str, platform: str = "", max_retries: int = 3, board_config: dict = None) -> Dict:
        """
        Handles board-specific chat: LLM interprets user input, generates board command(s),
        sends to board, parses response, and explains to user.
        Supports SSH (Raspberry Pi) and serial/arduino-cli (Arduino/ESP32).
        """
        if board_config is None:
            board_config = {}
        try:
            conn_type = board_config.get("type", "")
            serial_port = board_config.get("port", "")
            fqbn = board_config.get("fqbn", "")

            # Build connection context for the LLM
            if conn_type == "ssh":
                conn_hint = "Connection: SSH to Raspberry Pi. Use standard Linux shell commands (e.g. uptime, vcgencmd measure_temp, python3 script.py)."
            elif conn_type == "serial":
                port_hint = f" on port {serial_port}" if serial_port else ""
                fqbn_hint = f" (FQBN: {fqbn})" if fqbn else ""
                conn_hint = (
                    f"Connection: USB serial{port_hint}{fqbn_hint}. "
                    f"Use arduino-cli commands for compile/upload (e.g. arduino-cli compile, arduino-cli upload, arduino-cli monitor). "
                    f"For direct serial communication, just provide the text to send."
                )
            else:
                conn_hint = "Connection: Local shell. Use shell commands."

            # 1. LLM: Generate the exact shell command for the user request
            cmd_state = ProjectState(
                messages=[
                    SystemMessage(content=get_system_prompt(platform)),
                    HumanMessage(content=(
                        f"Platform: {platform}\n"
                        f"You are connected to a {platform} board.\n"
                        f"{conn_hint}\n"
                        f"User request: {user_input}\n"
                        f"Reply ONLY with the exact shell command to fulfill the user request, tailored to the detected board. No explanation, no output, no formatting. For example: uptime"
                    ))
                ],
                platform=platform,
                requirements=user_input,
                current_step="processing",
                iteration_count=0
            )
            cmd_result = await asyncio.get_event_loop().run_in_executor(None, self.graph.invoke, cmd_state)
            cmd_message = cmd_result["messages"][-1]
            user_command = self._get_content_as_string(getattr(cmd_message, 'content', str(cmd_message))).strip().split('\n')[0]

            # 2. Send user command to board (platform-specific logic)
            def run_command(cmd):
                conn_type = board_config.get("type", "")

                # --- SSH (Raspberry Pi) ---
                if conn_type == "ssh" and board_config.get("host") and board_config.get("user"):
                    try:
                        ssh = paramiko.SSHClient()
                        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        connect_kwargs = {
                            "hostname": board_config["host"],
                            "username": board_config["user"],
                            "port": board_config.get("port", 22),
                            "timeout": 10,
                        }
                        if board_config.get("key_file"):
                            connect_kwargs["key_filename"] = board_config["key_file"]
                        elif board_config.get("password"):
                            connect_kwargs["password"] = board_config["password"]
                        ssh.connect(**connect_kwargs)
                        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
                        output = stdout.read().decode().strip() or stderr.read().decode().strip()
                        ssh.close()
                        return output
                    except Exception as e:
                        return f"[SSH error: {e}]"

                # --- Serial / arduino-cli (Arduino, ESP32) ---
                elif conn_type == "serial":
                    serial_port = board_config.get("port", "")
                    baud = board_config.get("baud", 9600)
                    fqbn = board_config.get("fqbn", "")

                    # If command looks like an arduino-cli command, run it directly
                    if cmd.strip().startswith("arduino-cli"):
                        # Inject port/fqbn if not already in command
                        if serial_port and "--port" not in cmd:
                            cmd += f" --port {serial_port}"
                        if fqbn and "--fqbn" not in cmd:
                            cmd += f" --fqbn {fqbn}"
                        try:
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                            return result.stdout.strip() or result.stderr.strip()
                        except Exception as e:
                            return f"[arduino-cli error: {e}]"

                    # For serial monitor / reading output
                    if serial_port:
                        try:
                            import serial
                            ser = serial.Serial(serial_port, baud, timeout=3)
                            # Send command to board
                            ser.write((cmd + "\n").encode())
                            import time
                            time.sleep(1)
                            output = ser.read(ser.in_waiting or 256).decode(errors="replace").strip()
                            ser.close()
                            return output if output else "[No response from board]"
                        except Exception as e:
                            return f"[Serial error: {e}]"
                    return "[No serial port configured]"

                # --- Fallback: local shell ---
                else:
                    try:
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                        output = result.stdout.strip() or result.stderr.strip()
                        return output
                    except Exception as e:
                        return f"[Error running command: {e}]"

            user_command_output = run_command(user_command)

            # 3. LLM: Explain the board's output to the user
            explain_state = ProjectState(
                messages=[
                    SystemMessage(content=get_system_prompt(platform)),
                    HumanMessage(content=f"User asked: {user_input}\nCommand sent: {user_command}\nBoard output: {user_command_output}\nFirst give the output to the user. For example, if the output for temperature is 56 degree C, then output, the temperature of your respective board is 56-degress Celcius, or if the user told to execute some piece of code then output, code executed successfully and the result obtained is whatever the code's actual output will be. After displaying the results, explain it to the user in a friendly way.")
                ],
                platform=platform,
                requirements=user_input,
                current_step="processing",
                iteration_count=0
            )
            explain_result = await asyncio.get_event_loop().run_in_executor(None, self.graph.invoke, explain_state)
            explain_message = explain_result["messages"][-1]
            response_content = self._get_content_as_string(getattr(explain_message, 'content', str(explain_message)))
            return {
                "success": True,
                "response": response_content,
                "platform": platform,
                "timestamp": datetime.now().isoformat(),
                "user_command": user_command,
                "user_command_output": user_command_output
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate_project(self, platform: str, requirements: str, project_name: str) -> Dict:
        try:
            code_result = await self.process_request(f"Generate {platform} code for: {requirements}", platform)
            if not code_result["success"]:
                return code_result

            doc_result = await self.process_request(
                f"Create markdown documentation for {project_name}: {requirements}", platform
            )

            project_dir = self.tools_instance.knowledge_base_path / "projects" / project_name
            project_dir.mkdir(parents=True, exist_ok=True)

            # Save files
            (project_dir / "project.json").write_text(json.dumps({
                "name": project_name, "platform": platform, "requirements": requirements,
                "timestamp": datetime.now().isoformat()
            }, indent=2))
            
            (project_dir / "README.md").write_text(doc_result.get("response", ""))
            
            code = self._extract_code_from_response(code_result["response"])
            ext = ".ino" if platform.lower() in ["arduino", "esp32"] else ".py"
            if code:
                (project_dir / f"{project_name}{ext}").write_text(code)

            return {"success": True, "project_name": project_name, "project_path": str(project_dir),
                    "files_created": ["project.json", "README.md", f"{project_name}{ext}"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_code_from_response(self, response) -> str:
        if not response:
            return ""
        if isinstance(response, list):
            response = '\n'.join(str(item) for item in response)
        if not isinstance(response, str):
            response = str(response)

        for pattern in [r'```(?:cpp|c\+\+|arduino|ino)\n(.*?)```', r'```python\n(.*?)```', r'```\n(.*?)```']:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        return ""
