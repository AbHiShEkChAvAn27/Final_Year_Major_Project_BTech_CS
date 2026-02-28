"""
CLI Interface for Embedded Systems AI Agent
"""

import asyncio
from datetime import datetime
from pathlib import Path
from src.agent import EmbeddedSystemsAgent
from src.hardware.detector import detect_all_boards, get_platform_from_boards, format_board_summary


class EmbeddedSystemsCLI:
    """Enhanced CLI interface with LangGraph integration"""

    def __init__(self, api_key: str):
        self.agent = EmbeddedSystemsAgent(api_key)
        self.current_platform = ""
        self.detected_boards = []
        self.active_board = None
        self.board_config = {}
        self.session_history = []

    def _auto_detect_boards(self):
        """Scan for connected boards and set platform automatically."""
        print("\n🔍 Scanning for connected boards...")
        self.detected_boards = detect_all_boards()
        if self.detected_boards:
            self.current_platform = get_platform_from_boards(self.detected_boards)
            print(f"✅ Found {len(self.detected_boards)} board(s):")
            for i, b in enumerate(self.detected_boards, 1):
                port = b.get('port', 'N/A')
                print(f"   {i}. {b['board']} — {b['platform']} on {port}")
            idx = 0
            if len(self.detected_boards) > 1:
                try:
                    choice = input(f"Select board [1-{len(self.detected_boards)}] (default 1): ").strip()
                    idx = int(choice) - 1 if choice else 0
                    idx = max(0, min(idx, len(self.detected_boards) - 1))
                except ValueError:
                    idx = 0
            self.active_board = self.detected_boards[idx]
            self.current_platform = self.active_board["platform"]
            print(f"🎯 Active platform: {self.current_platform}")

            # Prompt for board-specific credentials
            self._configure_board_credentials()
        else:
            print("⚠️  No boards detected. Connect a board via USB and type 'rescan'.")
            self.current_platform = ""
            self.active_board = None
            self.board_config = {}

    def _configure_board_credentials(self):
        """Prompt for credentials based on detected board type."""
        if not self.active_board:
            self.board_config = {}
            return

        platform = self.active_board["platform"]

        if platform == "raspberry_pi":
            print("\n🔑 Raspberry Pi SSH credentials required:")
            host = input("   Host/IP [localhost]: ").strip() or "localhost"
            user = input("   Username [pi]: ").strip() or "pi"
            password = input("   Password (leave blank if using SSH key): ").strip()
            key_file = ""
            if not password:
                key_file = input("   SSH Key Path [~/.ssh/id_rsa]: ").strip() or "~/.ssh/id_rsa"
            port = input("   SSH Port [22]: ").strip() or "22"
            self.board_config = {
                "type": "ssh",
                "host": host, "user": user,
                "password": password, "key_file": key_file,
                "port": int(port),
            }
            print("✅ SSH config saved.")

        elif platform in ["arduino", "esp32"]:
            serial_port = self.active_board.get("port", "")
            fqbn = self.active_board.get("fqbn", "")
            print(f"\n🔌 Serial config: port={serial_port}  baud=115200  fqbn={fqbn or 'N/A'}")
            self.board_config = {
                "type": "serial",
                "port": serial_port,
                "baud": 115200,
                "fqbn": fqbn,
            }
        else:
            self.board_config = {}

    async def run_interactive_session(self):
        """Run enhanced interactive session"""
        print("🤖 Embedded Systems AI Agent")
        print("=" * 50)

        # Auto-detect boards on startup
        self._auto_detect_boards()

        print("\nCommands: chat, generate, project, search, knowledge, tools, rescan, history, help, quit")
        print("=" * 50)

        while True:
            try:
                command = input(f"\n[{self.current_platform or 'no board'}] > ").strip().lower()

                if command == "quit":
                    print("👋 Goodbye!")
                    break
                elif command == "chat":
                    await self._handle_chat()
                elif command == "generate":
                    await self._handle_generate()
                elif command == "project":
                    await self._handle_project()
                elif command == "search":
                    await self._handle_search()
                elif command == "knowledge":
                    await self._handle_knowledge()
                elif command == "tools":
                    self._show_tools()
                elif command == "rescan":
                    self._auto_detect_boards()
                elif command == "history":
                    self._show_history()
                elif command == "help":
                    self._show_help()
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def _handle_chat(self):
        question = input("Ask me anything: ").strip()
        if not question:
            return
        print("🤔 Thinking...")
        result = await self.agent.process_request(question, self.current_platform, board_config=self.board_config)
        if result["success"]:
            print(f"\n💡 {result['response']}")
            self.session_history.append({"type": "chat", "question": question, "timestamp": result['timestamp']})
        else:
            print(f"❌ {result['error']}")

    async def _handle_generate(self):
        if not self.current_platform:
            print("⚠️  No board detected. Connect a board and type 'rescan'.")
            return
        platform = self.current_platform
        requirements = input("Describe what you want: ").strip()
        if not requirements:
            return
        print("⚡ Generating...")
        result = await self.agent.process_request(f"Generate {platform} code for: {requirements}", platform)
        if result["success"]:
            print(f"\n✅ Generated:\n{'='*50}\n{result['response']}\n{'='*50}")
            self.session_history.append({"type": "generate", "platform": platform, "timestamp": result['timestamp']})
        else:
            print(f"❌ {result['error']}")

    async def _handle_project(self):
        if not self.current_platform:
            print("⚠️  No board detected. Connect a board and type 'rescan'.")
            return
        platform = self.current_platform
        name = input("Project name: ").strip()
        requirements = input("Requirements: ").strip()
        if not name or not requirements:
            print("⚠️ Name and requirements required")
            return
        print("🏗️ Creating project...")
        result = await self.agent.generate_project(platform, requirements, name)
        if result["success"]:
            print(f"✅ Created: {result['project_path']}")
            print(f"📄 Files: {', '.join(result['files_created'])}")
        else:
            print(f"❌ {result['error']}")

    async def _handle_search(self):
        query = input("Search: ").strip()
        if query:
            print("🔍 Searching...")
            result = await self.agent.process_request(f"Search for: {query}")
            print(result.get('response', result.get('error', 'No results')))

    async def _handle_knowledge(self):
        file_path = input("File path (PDF/TXT): ").strip()
        if file_path:
            success = await self.agent.add_knowledge(file_path)
            print("✅ Added!" if success else "❌ Failed")

    def _show_tools(self):
        print("\n🔧 Available Tools:")
        tools = ["web_search", "component_lookup", "pinout_lookup", "code_template", "code_validator", "library_lookup", "file_operations"]
        for t in tools:
            print(f"  - {t}")

    def _show_history(self):
        if not self.session_history:
            print("📝 No history")
            return
        print(f"\n📝 History ({len(self.session_history)} items):")
        for item in self.session_history[-5:]:
            print(f"  [{item['type']}] {item.get('timestamp', '')}")

    def _show_help(self):
        print("""
🤖 Commands:
  chat      - Ask questions
  generate  - Generate code
  project   - Create full project
  search    - Web search
  knowledge - Add documents
  tools     - List tools
  rescan    - Re-detect connected boards
  history   - View history
  help      - Show this
  quit      - Exit
""")
