"""
Embedded Systems AI Agent - Streamlit Application
"""

import streamlit as st
import asyncio
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from src.agent import EmbeddedSystemsAgent
from src.ui import extract_mermaid, render_mermaid
from src.tools.power_estimator import power_profile_estimator_fn
from src.hardware.detector import detect_all_boards, get_platform_from_boards, format_board_summary

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Streamlit page config
st.set_page_config(page_title="Embedded Systems AI Agent", layout="wide", page_icon="🤖")
st.title("🤖 Embedded Systems AI Agent")

# Session state
for key, default in [("agent", None), ("platform", ""), ("projects_list", []), ("messages", []), ("detected_boards", []), ("selected_board_idx", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default


def init_agent(platform):
    if not API_KEY:
        st.error("API key not found! Set NVIDIA_API_KEY or GEMINI_API_KEY in .env")
        return False
    try:
        st.session_state["agent"] = EmbeddedSystemsAgent(API_KEY)
        st.session_state["platform"] = platform
        st.success("Agent initialized! 🚀")
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False


def handle_rate_limit(msg):
    st.warning(f"⏳ {msg}")
    st.info("💡 Free tier: 20 requests/day. Consider upgrading.")


# --- Board Auto-Detection ---
def refresh_boards():
    """Scan for connected boards and update session state."""
    boards = detect_all_boards()
    st.session_state["detected_boards"] = boards
    if boards:
        st.session_state["platform"] = get_platform_from_boards(boards)
    else:
        st.session_state["platform"] = ""

# Auto-detect on first load
if not st.session_state["detected_boards"]:
    refresh_boards()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Detected Boards Section ---
    st.subheader("🔌 Connected Boards")
    if st.button("🔄 Refresh Boards"):
        refresh_boards()
        st.rerun()

    boards = st.session_state["detected_boards"]
    if boards:
        st.markdown(format_board_summary(boards))
        if len(boards) > 1:
            board_labels = [f"{b['board']} ({b.get('port','')})" for b in boards]
            selected_idx = st.selectbox("Active board", range(len(board_labels)), format_func=lambda i: board_labels[i], key="selected_board_idx")
            platform = boards[selected_idx]["platform"]
        else:
            platform = boards[0]["platform"]
        st.success(f"Platform: **{platform}**")
    else:
        st.warning("No boards detected. Connect a board via USB and click **Refresh Boards**.")
        platform = ""

    # --- Board-specific config (auto-set for Arduino/ESP32, SSH asked in Chat with Board tab) ---
    if platform in ["arduino", "esp32"] and boards:
        active_board = boards[st.session_state.get("selected_board_idx", 0)] if len(boards) > 1 else boards[0]
        serial_port = active_board.get("port", "")
        fqbn = active_board.get("fqbn", "")
        baud = 115200  # sensible default
        st.info(f"🔌 Auto-detected: **{serial_port}** @ {baud} baud" + (f"  •  FQBN: `{fqbn}`" if fqbn else ""))
        st.session_state["board_config"] = {
            "type": "serial",
            "port": serial_port, "baud": baud, "fqbn": fqbn,
        }
    elif platform == "raspberry_pi":
        # SSH credentials will be entered in the Chat with Board tab
        if "board_config" not in st.session_state or st.session_state["board_config"].get("type") != "ssh":
            st.session_state["board_config"] = {"type": "ssh"}
    else:
        st.session_state["board_config"] = {}

    if st.button("Initialize Agent"):
        init_agent(platform)
    
    st.markdown("---")
    st.subheader("📁 Projects")
    projects_path = Path("./knowledge_base/projects")
    projects_list = [p.name for p in projects_path.iterdir() if p.is_dir()] if projects_path.exists() else []
    st.session_state["projects_list"] = projects_list
    selected_project = st.selectbox("Select project", [""] + projects_list)

if not st.session_state["agent"]:
    st.warning("Please initialize the agent first.")
    st.stop()

agent = st.session_state["agent"]
st.session_state["platform"] = platform

# Tabs
tab_chat, tab_generate, tab_project, tab_knowledge, tab_history, tab_board = st.tabs(
    ["💬 Chat", "⚡ Generate Code", "🏗️ Project Builder", "📚 Knowledge Base", "📝 Debug", "🛠️ Chat with Board"]
)

with tab_chat:
    st.subheader("💬 Embedded Systems Assistant")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about pinouts, protocols, sensors..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = asyncio.run(agent.process_request(prompt, platform))
                if result["success"]:
                    st.markdown(result["response"])
                    if mermaid := extract_mermaid(result["response"]):
                        st.markdown("**Wiring Diagram:**")
                        render_mermaid(mermaid)
                    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                else:
                    if "rate limit" in result["error"].lower():
                        handle_rate_limit(result["error"])
                    else:
                        st.error(result["error"])

with tab_generate:
    st.subheader("⚡ Quick Code Snippet")
    requirements = st.text_area("What should the code do?", placeholder="e.g. Read DHT22 sensor")
    if st.button("Generate Code", key="gen"):
        if requirements:
            with st.spinner("Generating..."):
                result = asyncio.run(agent.process_request(f"Generate {platform} code for: {requirements}", platform))
            if result["success"]:
                lang = "cpp" if platform in ["arduino", "esp32"] else "python"
                st.code(result["response"], language=lang)
                if mermaid := extract_mermaid(result["response"]):
                    st.markdown("**Wiring Diagram:**")
                    render_mermaid(mermaid)
                try:
                    st.markdown("**⚡ Power Estimate:**")
                    st.json(power_profile_estimator_fn(result["response"], platform))
                except:
                    pass
                st.download_button("📥 Download", result["response"], f"code.{'ino' if lang=='cpp' else 'py'}")
            else:
                st.error(result["error"])

with tab_project:
    st.subheader("🏗️ Full Project Workspace")
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Project Name", placeholder="Smart_Garden")
    with col2:
        st.info(f"Platform: {platform}")
    
    p_reqs = st.text_area("Requirements", placeholder="List sensors and logic...")
    if st.button("Build Project", key="proj"):
        if p_name and p_reqs:
            with st.status("🏗️ Building...", expanded=True) as status:
                result = asyncio.run(agent.generate_project(platform, p_reqs, p_name))
                if result["success"]:
                    status.update(label="✅ Built!", state="complete")
                    proj_dir = Path(f"./knowledge_base/projects/{p_name}")
                    if proj_dir.exists():
                        shutil.make_archive(p_name, 'zip', proj_dir)
                        with open(f"{p_name}.zip", "rb") as f:
                            st.download_button(f"📦 Download {p_name}.zip", f, f"{p_name}.zip")
                else:
                    status.update(label="❌ Failed", state="error")
                    st.error(result["error"])

with tab_knowledge:
    st.subheader("📚 Datasheet Indexer")
    uploaded = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    if uploaded:
        temp_path = Path(f"temp_{uploaded.name}")
        temp_path.write_bytes(uploaded.getbuffer())
        with st.spinner("Analyzing..."):
            success = asyncio.run(agent.add_knowledge(str(temp_path)))
        st.success(f"Indexed {uploaded.name}!") if success else st.error("Failed")
        temp_path.unlink(missing_ok=True)

with tab_history:
    st.subheader("📝 Session Debug")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state['board_messages'] = []
        st.rerun()
    st.json({"Platform": platform, "Projects": st.session_state["projects_list"]})

with tab_board:
    st.subheader("🛠️ Chat with Board")

    # --- Show SSH config form inside this tab when RPi is detected ---
    if platform == "raspberry_pi":
        with st.expander("🔑 Raspberry Pi SSH Configuration", expanded=not st.session_state.get("board_config", {}).get("host")):
            ssh_host = st.text_input("RPi Host (IP or hostname)", key="ssh_host")
            ssh_user = st.text_input("RPi Username", value="pi", key="ssh_user")
            ssh_pass = st.text_input("RPi Password", type="password", key="ssh_pass")
            ssh_key = st.text_input("SSH Key Path (optional)", placeholder="~/.ssh/id_rsa", key="ssh_key")
            ssh_port = st.number_input("SSH Port", value=22, min_value=1, max_value=65535, key="ssh_port")
            st.session_state["board_config"] = {
                "type": "ssh",
                "host": ssh_host, "user": ssh_user,
                "password": ssh_pass, "key_file": ssh_key,
                "port": int(ssh_port),
            }
            if ssh_host and ssh_user:
                st.success(f"SSH configured: {ssh_user}@{ssh_host}:{ssh_port}")
            else:
                st.warning("Enter SSH credentials above to chat with your Raspberry Pi.")

    if 'board_messages' not in st.session_state:
        st.session_state['board_messages'] = []

    for msg in st.session_state['board_messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    board_prompt = st.chat_input("Send a command or ask about the board...")
    if board_prompt:
        with st.chat_message("user"):
            st.markdown(board_prompt)
        st.session_state['board_messages'].append({"role": "user", "content": board_prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                board_config = st.session_state.get("board_config", {})
                result = asyncio.run(agent.process_request(board_prompt, platform, mode="board", board_config=board_config))
                if result["success"]:
                    st.markdown(result["response"])
                    st.session_state['board_messages'].append({"role": "assistant", "content": result["response"]})
                else:
                    st.error(result["error"])

if selected_project:
    st.divider()
    st.subheader(f"📂 {selected_project}")
    proj_dir = projects_path / selected_project
    if (readme := proj_dir / "README.md").exists():
        with st.expander("📄 Documentation", expanded=True):
            st.markdown(readme.read_text())
    for ext in [".py", ".ino", ".cpp"]:
        if (code_file := proj_dir / f"{selected_project}{ext}").exists():
            with st.expander("💻 Source Code"):
                st.code(code_file.read_text(), language="cpp" if ext != ".py" else "python")