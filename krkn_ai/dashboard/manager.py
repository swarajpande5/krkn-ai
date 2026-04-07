import os
import sys
import subprocess
from krkn_ai.utils.logger import get_logger

class DashboardManager:
    @staticmethod
    def start(output_dir: str, port: int, status: str, background: bool = True):
        logger = get_logger(__name__)
        dashboard_dir = os.path.dirname(__file__)
        actual_output = os.path.abspath(output_dir if output_dir else "./")
        
        try:
            import streamlit
        except ImportError:
            logger.error("Monitoring dependencies not found. Please install them using 'pip install krkn-ai[monitor]'.")
            sys.exit(1)
            
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            os.path.join(dashboard_dir, "app.py"),
            "--server.port",
            str(port),
            "--",
            "--output-dir",
            actual_output,
        ]
        
        try:
            if background:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Dashboard running at http://localhost:{port}")
                return process
            else:
                subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            logger.info("Monitoring dashboard stopped.")
        except Exception as e:
            logger.error(f"Failed to start monitoring dashboard: {e}")
            return None
