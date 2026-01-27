use managed_process::{BackendController, ProcessProtocol, RunBackendProcess};
use std::process::Stdio;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    spawn,
};
use tracing::{info,error};

mod comfyuiconfig;
pub use comfyuiconfig::{AttnSetting, ComfyUiConfig, ComfyUiConfigArgs, VRamSetting};

pub type ComfyUiProtocol = ProcessProtocol<ComfyUiConfig>;
pub type ComfyUiBackendController = BackendController<ComfyUiConfig>;

pub struct ComfyUiBackend {
    pub listen: String,
    pub port: u16,
    pub comfyui_setup_sh: String,
    pub comfyui_main_py: String,
    pub comfyui_execdir: String,
}

impl RunBackendProcess for ComfyUiBackend {
    type ProcessConfig = ComfyUiConfig;

    fn run_backend_process(
        &self,
        process_config: Self::ProcessConfig,
        cancel_receiver: tokio::sync::oneshot::Receiver<bool>,
        notifier: tokio::sync::mpsc::Sender<ProcessProtocol<Self::ProcessConfig>>,
    ) {
        // run normal-setup
        let mut cmd = std::process::Command::new(format!("./{}", self.comfyui_setup_sh).as_str());
        cmd.current_dir(&self.comfyui_execdir);
        cmd.spawn().unwrap().wait().unwrap();

        // prepare comfyui-command:
        //let mut cmd = Command::new("uv");
        //cmd.arg("run");

        let mut cmd = Command::new(".venv/bin/python3");
        cmd.current_dir(&self.comfyui_execdir);
        cmd.arg(self.comfyui_main_py.as_str());

        // set environment variables
        process_config.apply_env(&mut cmd);

        // set params
        cmd.arg("--listen");
        cmd.arg(&self.listen);

        cmd.arg("--port");
        cmd.arg(self.port.to_string());

        // provide std-streams
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        cmd.kill_on_drop(true);

        // spawn process
        let mut proc_handle = cmd.spawn().unwrap();

        // spawn std-err observing task
        let stderr = proc_handle.stderr.take().unwrap();
        let notifier_cloned = notifier.clone();
        spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            loop {
                if let Ok(Some(line)) = lines.next_line().await
                    && line.contains("To see the GUI go to: http")
                {
                    notifier_cloned
                        .send(ComfyUiProtocol::ProcessStarted)
                        .await
                        .unwrap();
                }
            }
        });

        spawn(async move {
            tokio::select! {
                s = proc_handle.wait() => {
                    let exit_status = s.unwrap();
                    if exit_status.success() {
                        info!("comfyui-process ended successfully");
                        notifier.send(ComfyUiProtocol::ProcessFinished(None)).await.unwrap();
                    } else {
                        error!("comfyui-process ended unsuccessfully with error exit_status {exit_status}");
                        notifier.send(ComfyUiProtocol::ProcessFinished(Some(exit_status))).await.unwrap();
                    }
                },
                _ = cancel_receiver => {
                    info!("killing comfyui-process");
                    proc_handle.kill().await.unwrap();
                    notifier.send(ComfyUiProtocol::ProcessFinished(None)).await.unwrap();
                }
            }
        });
    }
}
